"""Serve a small model with selective capture into an in-process consumer.

Run by ``test_selective_capture_gpu.py`` in a subprocess:

    python -m tests.vllm_selective_runner MODE MODEL OUT_JSON

``MODE`` is ``gather`` (per-HookPoint row tables, prefill selection and node
toggling together), ``toggle`` (node toggling alone), ``heads`` (in-graph
heads), ``fused_toggle`` (heads and node toggling without prefill selection,
only the head key enabled) or ``returned`` (``gather`` with
``dmx_resid_pre_read=returned``).
The runner checks what the consumer received against an HF reference
computed on CPU after vLLM shuts down, and writes a JSON verdict.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process sink factory
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

L1, L2 = 3, 7
PROMPTS = ["The capital of France is", "One, two, three, four,",
           "In a distant galaxy"]
MAX_TOKENS = 6
RESID_PRE, RESID_PRE_HEAD = 0, 24


class Sink:
    def __init__(self, queue):
        self.ring_sink = queue
        self.dmx_batch_submit = True
        self.finalized = []

    def finalize_request(self, rid, valid=True):
        self.finalized.append((rid, bool(valid)))


def _config(mode):
    config = {
        "dmx_hook_selection": "resid_pre",
        "dmx_ring_payload_mb": 64,
        "dmx_ring_pinned_mb": 64,
        "dmx_ring_task_entries": 4096,
        "dmx_drain_flush_timeout_us": 2000,
        "dmx_require_sink": True,
    }
    if mode in ("gather", "returned", "heads"):
        config.update(dmx_per_hook_request_gather=True, dmx_prefill_selection=True)
    if mode in ("gather", "returned", "toggle"):
        config.update(dmx_node_toggle=True, dmx_lazy_toggle=True,
                      dmx_enabled_hooks=f"{RESID_PRE}:{L1},{RESID_PRE}:{L2}")
    if mode == "toggle":
        config["dmx_enabled_hooks"] = f"{RESID_PRE}:{L1}"
    if mode == "returned":
        config["dmx_resid_pre_read"] = "returned"
    if mode in ("heads", "fused_toggle"):
        config.update(dmx_inline_head_slots=2, dmx_hook_layers=f"{L1},{L2}")
    if mode == "fused_toggle":
        config.update(dmx_per_hook_request_gather=True, dmx_node_toggle=True,
                      dmx_lazy_toggle=True, dmx_enabled_hooks=f"{RESID_PRE_HEAD}:{L1}")
    return config


def main(mode, model, out_path):
    import torch
    from vllm import LLM, SamplingParams

    from dmi.transport import native as ne
    from dmi_vllm_integration.sinks import clear_sink_factory, register_sink_factory

    queue = ne.BatchQueue(256 << 20)
    sink = Sink(queue)
    register_sink_factory(lambda: sink)
    llm = LLM(model=model, worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
              dtype="bfloat16", max_model_len=256, max_num_seqs=4,
              gpu_memory_utilization=float(os.environ.get("DMI_VLLM_TEST_GPU_MEMORY", "0.2")),
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4],
                                  "cudagraph_mode": "FULL_AND_PIECEWISE"},
              additional_config=_config(mode), seed=0)
    clear_sink_factory()
    torch.manual_seed(0)
    weight = torch.randn(1024).tolist()
    if mode in ("gather", "returned"):
        llm.collective_rpc("dmx_set_hook_request_selection", args=(
            {"0": [(RESID_PRE, L1)], "1": [(RESID_PRE, L1), (RESID_PRE, L2)]}, [],
            {"1": {(RESID_PRE, L2): [(2, 5)]}}))
    elif mode in ("heads", "fused_toggle"):
        llm.collective_rpc("dmx_inline_register", args=([
            {"hook": [RESID_PRE_HEAD, L1], "slot": 0, "weight": weight, "bias": 0.25}],))
        llm.collective_rpc("dmx_set_hook_request_selection", args=(
            {"0": [(RESID_PRE, L1), (RESID_PRE_HEAD, L1)]}, []))
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
    completion = llm.collective_rpc("dmx_finalize_requests",
                                    args=([o.request_id for o in outputs],))[0]
    state = llm.collective_rpc("dmx_request_mask_state")[0]
    llm.collective_rpc("stop_monitoring")
    stats = llm.collective_rpc("dmx_sink_stats")[0]
    tokens = {o.request_id: list(o.prompt_token_ids) + list(o.outputs[0].token_ids)
              for o in outputs}
    prompts = {o.request_id: len(o.prompt_token_ids) for o in outputs}
    del llm

    # (request, act, layer) -> {position: row}; step -> layers delivered.
    rows = defaultdict(dict)
    step_layers = defaultdict(set)
    while True:
        batches = queue.drain_batches(256, 50)
        if not batches:
            break
        for batch in batches:
            _, _, act, layer, _, step, _, tensor, rids, segments = batch[:10]
            step_layers[step].add((act, layer))
            # One segments row per request: (offset, start, end, position).
            for rid, (offset, start, end, _) in zip(rids, segments.tolist()):
                for k in range(end - start):
                    rows[(rid, act, layer)][start + k] = tensor[offset + k].float()

    verdict = {"mode": mode, "finalized": sorted(sink.finalized),
               "completion": completion, "state": {
        k: state[k] for k in ("hook_table_uploads", "hook_table_reuses", "gather_host_skips")
        if k in state}, "submit_exceptions": stats["submit_exceptions"],
        "captured": sorted([rid, act, layer, sorted(pos)] for (rid, act, layer), pos in rows.items()),
        "step_layer_sets": sorted({tuple(sorted(l for _, l in v)) for v in step_layers.values()})}

    # HF reference on CPU: hidden_states[L] is resid_pre of layer L.
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    worst = 1.0
    head_err = 0.0
    with torch.no_grad():
        for rid, ids in tokens.items():
            hidden = ref(torch.tensor([ids]), output_hidden_states=True).hidden_states
            for (r, act, layer), positions in rows.items():
                if r != rid or act != "blocks.hook_resid_pre":
                    continue
                for pos, row in positions.items():
                    cos = torch.nn.functional.cosine_similarity(
                        row, hidden[layer][0, pos], dim=0).item()
                    worst = min(worst, cos)
    for (rid, act, layer), positions in rows.items():
        if act != "blocks.hook_resid_pre_head":
            continue
        source = rows[(rid, "blocks.hook_resid_pre", layer)]
        w = torch.tensor(weight)
        for pos, score in positions.items():
            if pos not in source:
                continue
            expect = float(source[pos] @ w) + 0.25
            head_err = max(head_err, abs(score[0].item() - expect) / max(1.0, abs(expect)))
            verdict.setdefault("free_slot_nan", True)
            verdict["free_slot_nan"] &= bool(torch.isnan(score[1]))
    verdict.update(min_cosine=worst, head_rel_err=head_err, prompts=prompts,
                   lengths={rid: len(ids) for rid, ids in tokens.items()})
    with open(out_path, "w") as stream:
        json.dump(verdict, stream)


if __name__ == "__main__":
    main(*sys.argv[1:4])
