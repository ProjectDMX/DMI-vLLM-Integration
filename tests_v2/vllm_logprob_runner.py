"""Run a vLLM model and save full-vocab logprobs to disk as tensors.

Two modes:
  --ref : load ref model (architecture remap + REF_CONFIG buffers)
  default: load original model (stock vLLM)

Usage:
    python -m tests_v2.vllm_logprob_runner --output /tmp/logprobs_orig.pt
    REF_CONFIG=/tmp/ref_config.json python -m tests_v2.vllm_logprob_runner --output /tmp/logprobs_ref.pt --ref
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch
from vllm.v1.worker.gpu_worker import Worker

from dmi_vllm_integration.v2.worker import DMXV2GPUWorker
from tests.oracles import register_oracle_models


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B",
}

_ARCH_REMAP = {
    "GPT2LMHeadModel": "DMIGPT2RefLMHeadModel",
    "Qwen2MoeForCausalLM": "DMIQwen2MoeRefForCausalLM",
    "Qwen3ForCausalLM": "DMIQwen3RefForCausalLM",
    "LlamaForCausalLM": "DMILlamaRefForCausalLM",
}


class RefLogprobWorker(Worker):
    """Minimal worker that remaps architecture to ref variant."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        register_oracle_models()
        hf_cfg = self.vllm_config.model_config.hf_config
        archs = getattr(hf_cfg, "architectures", [])
        new_archs = [_ARCH_REMAP.get(a, a) for a in archs]
        hf_cfg.architectures = new_archs
        super().load_model(load_dummy_weights=load_dummy_weights)


class _LogitCaptureMixin:
    """Capture only real-request logits, after engine warmup has completed."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._e2e_capture_logits = False
        self._e2e_raw_logits = []
        model = self.model_runner.model
        original_compute_logits = model.compute_logits

        def capture_compute_logits(*args, **kwargs):
            logits = original_compute_logits(*args, **kwargs)
            if self._e2e_capture_logits:
                self._e2e_raw_logits.append(
                    logits.detach().to(device="cpu").contiguous().clone()
                )
            return logits

        model.compute_logits = capture_compute_logits

    def start_logit_capture(self) -> None:
        self._e2e_raw_logits.clear()
        self._e2e_capture_logits = True

    def describe_model_runner(self) -> str:
        runner_type = type(self.model_runner)
        return f"{runner_type.__module__}.{runner_type.__qualname__}"

    def dump_raw_logits(self) -> int:
        self._e2e_capture_logits = False
        output = os.environ.get("E2E_RAW_LOGITS_OUTPUT")
        if not output:
            raise RuntimeError("E2E_RAW_LOGITS_OUTPUT is not configured")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tuple(self._e2e_raw_logits), output_path)
        return len(self._e2e_raw_logits)


class StockLogitCaptureWorker(_LogitCaptureMixin, Worker):
    """Stock vLLM worker with an output-only test tap."""


class MonitoredLogitCaptureWorker(_LogitCaptureMixin, DMXV2GPUWorker):
    """DMI worker with the same output-only test tap."""


def main():
    register_oracle_models()
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--ref", action="store_true",
                   help="Use ref model (architecture remap + REF_CONFIG)")
    p.add_argument("--monitored", action="store_true",
                   help="Use DMXGPUWorker (ring transport hooks)")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed info about stored logprob tensors")
    p.add_argument("--random-prompts", action="store_true",
                   help="Use random integers in prompt template")
    p.add_argument("--math", action="store_true",
                   help="Use math sequence prompts")
    p.add_argument("--chat", action="store_true",
                   help="Use realistic conversation prompts")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for --random-prompts")
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")
    hook_selection = os.environ.get("DMX_HOOK_SELECTION", "vllm-full")
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))

    prompts_json = os.environ.get("E2E_PROMPTS_JSON")
    if prompts_json:
        prompts = json.loads(prompts_json)
        if not isinstance(prompts, list) or not prompts or not all(
            isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError("E2E_PROMPTS_JSON must be a non-empty string list")
    elif args.random_prompts:
        import random
        seed = args.seed if args.seed is not None else int(os.environ.get("E2E_SEED", "42"))
        rng = random.Random(seed)
        numbers = [rng.randint(1, 100000) for _ in range(num_prompts)]
        prompts = _prompts_from_numbers(numbers, args)
    else:
        prompts = _prompts_from_numbers(
            list(range(1, num_prompts + 1)), args
        )
    if os.environ.get("E2E_PRINT_PROMPTS", "0") == "1":
        for i, p in enumerate(prompts):
            print(f"[vllm_logprob_runner] prompt[{i}]: {p!r}", flush=True)

    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))
    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(
            os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "512")),
        max_logprobs=-1,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )
    raw_logits_output = os.environ.get("E2E_RAW_LOGITS_OUTPUT")
    if args.ref:
        kwargs["worker_cls"] = "tests_v2.vllm_logprob_runner.RefLogprobWorker"
    elif args.monitored:
        kwargs["worker_cls"] = (
            "tests_v2.vllm_logprob_runner.MonitoredLogitCaptureWorker"
            if raw_logits_output
            else "dmi_vllm_integration.v2.worker.DMXV2GPUWorker"
        )
        kwargs["additional_config"] = {
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
        }
    elif raw_logits_output:
        kwargs["worker_cls"] = (
            "tests_v2.vllm_logprob_runner.StockLogitCaptureWorker"
        )

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=-1)
    if raw_logits_output:
        runner_types = llm.collective_rpc("describe_model_runner")
        print(
            f"[vllm_logprob_runner] Model runners: {runner_types}",
            flush=True,
        )
        llm.collective_rpc("start_logit_capture")
    outputs = llm.generate(prompts, params)
    if raw_logits_output:
        counts = llm.collective_rpc("dump_raw_logits")
        print(
            f"[vllm_logprob_runner] Captured raw-logit calls: {counts}",
            flush=True,
        )

    # Build dense logprob tensors: {prompt_idx: {token_ids, logprobs_tensor}}
    result = {}
    for i, o in enumerate(outputs):
        token_ids = list(o.outputs[0].token_ids)
        steps = o.outputs[0].logprobs  # list[dict[int, Logprob]]
        public_output = {
            "token_ids": token_ids,
            "text": o.outputs[0].text,
            "finish_reason": o.outputs[0].finish_reason,
            "stop_reason": o.outputs[0].stop_reason,
            "prompt_token_ids": list(o.prompt_token_ids),
        }
        if not steps:
            result[i] = {**public_output, "logprobs": None}
            continue

        # Determine vocab size from first step (logprobs=-1 returns all via gpu_input_batch)
        vocab_size = max(max(step.keys()) for step in steps) + 1
        entries_per_step = [len(step) for step in steps]
        print(f"  prompt[{i}]: {len(steps)} tokens, vocab={vocab_size}, "
              f"entries_per_step={min(entries_per_step)}-{max(entries_per_step)}",
              flush=True)
        if min(entries_per_step) < vocab_size:
            print(f"  WARNING: NOT full vocab! Expected {vocab_size}, got {min(entries_per_step)}",
                  flush=True)
        num_tokens = len(steps)
        logprob_tensor = torch.full((num_tokens, vocab_size), float("-inf"),
                                    dtype=torch.float32)
        for t, step in enumerate(steps):
            for tid, lp in step.items():
                logprob_tensor[t, tid] = lp.logprob

        result[i] = {**public_output, "logprobs": logprob_tensor}
        print(f"  prompt[{i}]: {num_tokens} tokens, vocab={vocab_size}")

    torch.save(result, args.output)
    print(f"[vllm_logprob_runner] Saved {len(result)} prompts to {args.output}",
          flush=True)

    if args.verbose:
        for i in sorted(result.keys()):
            r = result[i]
            tids = r["token_ids"]
            lp = r["logprobs"]
            if lp is not None:
                finite = torch.isfinite(lp).sum().item()
                print(f"  stored[{i}]: tokens={len(tids)} "
                      f"logprobs shape={list(lp.shape)} dtype={lp.dtype} "
                      f"finite={finite}/{lp.numel()} "
                      f"first_step_entries={len(lp[0][lp[0] > float('-inf')])}",
                      flush=True)
            else:
                print(f"  stored[{i}]: tokens={len(tids)} logprobs=None",
                      flush=True)

    # Flush DMI workers explicitly only for the monitored path.
    if args.monitored:
        try:
            llm.collective_rpc("stop_monitoring")
        except Exception:
            pass
    del llm
    torch.cuda.empty_cache()


def _prompts_from_numbers(numbers, args):
    if args.chat:
        return [
            f"The {n}th most spoken language in the world is"
            for n in numbers
        ]
    if args.math:
        return [
            f"Start from {n}, generate a sequence of numbers with gap 1:"
            for n in numbers
        ]
    return [f"The answer to question {n} is" for n in numbers]


if __name__ == "__main__":
    main()
