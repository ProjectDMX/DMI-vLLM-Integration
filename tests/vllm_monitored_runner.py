"""Standalone script: run vLLM with DMXGPUWorker (hooked model + ring transport).

Activations go to ClickHouse. Saves metadata to disk for the comparator.

Usage:
    python -m tests.vllm_monitored_runner --output-dir /tmp/vllm_mon
"""
import argparse
import json
import os
import re
import uuid

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch


_MODEL_ALIASES = {
    "gpt2": "gpt2",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.1-8B",
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args, _ = p.parse_known_args()

    from vllm import LLM, SamplingParams

    model_key = os.environ.get("E2E_MODEL", "gpt2")
    model_id = _MODEL_ALIASES.get(model_key, model_key)
    num_prompts = int(os.environ.get("E2E_NUM_PROMPTS", "8"))
    max_new_tokens = int(os.environ.get("E2E_MAX_NEW_TOKENS", "20"))
    enforce_eager = os.environ.get("E2E_ENFORCE_EAGER", "1") == "1"
    model_dtype = os.environ.get("E2E_DTYPE", "auto")
    ring_payload_mb = int(os.environ.get("E2E_RING_PAYLOAD_MB", "4096"))
    ring_pinned_mb = int(os.environ.get("E2E_RING_PINNED_MB", "4096"))
    hook_selection = os.environ.get("DMX_HOOK_SELECTION", "vllm-full")
    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_database = _identifier(os.environ.get("DMX_DB_DATABASE", "default"))
    db_table = _identifier(
        os.environ.get("DMX_DB_TABLE", f"dmi_vllm_test_{uuid.uuid4().hex[:12]}")
    )
    tp_size = int(os.environ.get("E2E_TP_SIZE", "1"))
    enable_ep = os.environ.get("E2E_ENABLE_EP", "0") == "1"
    all2all_backend = os.environ.get("E2E_ALL2ALL_BACKEND")
    moe_backend = os.environ.get("E2E_MOE_BACKEND")

    prompts = [f"The answer to question {i+1} is" for i in range(num_prompts)]

    import clickhouse_driver
    client = clickhouse_driver.Client(db_host, port=db_port)
    existing = client.execute(
        "SELECT count() FROM system.tables WHERE database=%(database)s "
        "AND name=%(table)s",
        {"database": db_database, "table": db_table},
    )[0][0]
    if existing:
        raise RuntimeError(f"refusing to reuse {db_database}.{db_table}")

    kwargs = dict(
        model=model_id,
        dtype=model_dtype,
        worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
        additional_config={
            "dmx_hook_selection": hook_selection,
            "dmx_ring_payload_mb": ring_payload_mb,
            "dmx_ring_pinned_mb": ring_pinned_mb,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
            "dmx_db_database": db_database,
            "dmx_db_table": db_table,
        },
        max_model_len=int(os.environ.get("E2E_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(
            os.environ.get("E2E_MAX_NUM_BATCHED_TOKENS", "512")),
        enforce_eager=enforce_eager,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.5")),
        tensor_parallel_size=tp_size,
    )
    if enable_ep:
        kwargs["enable_expert_parallel"] = True
    if all2all_backend:
        kwargs["all2all_backend"] = all2all_backend
    if moe_backend:
        kwargs["moe_backend"] = moe_backend
    cg_mode = os.environ.get("E2E_CUDAGRAPH_MODE")
    if cg_mode:
        kwargs["compilation_config"] = {"cudagraph_mode": cg_mode}
        print(f"[vllm_monitored_runner] cudagraph_mode={cg_mode}", flush=True)
    llm = LLM(**kwargs)

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    generated = {}
    for i, o in enumerate(outputs):
        generated[i] = len(o.outputs[0].token_ids)
        print(f"  prompt[{i}]: {generated[i]} tokens generated")

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "num_prompts": num_prompts,
            "max_new_tokens": max_new_tokens,
            "generated_tokens": generated,
            "db_host": db_host,
            "db_port": db_port,
            "db_database": db_database,
            "db_table": db_table,
        }, f)

    # Explicit per-worker flush+stop before LLM teardown. The implicit
    # path via DMXGPUWorker.shutdown() races vLLM's 8s deadline and warns
    # "Data may be incomplete." collective_rpc fans out to every TP rank.
    llm.collective_rpc("stop_monitoring")
    del llm
    torch.cuda.empty_cache()
    print(f"[vllm_monitored_runner] Done", flush=True)


if __name__ == "__main__":
    main()
