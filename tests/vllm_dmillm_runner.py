"""Subprocess runner: DMILLM offline -> read each RequestOutput's per-request
.dmi_internal back from ClickHouse, write pass/fail to a result file.

Run as a subprocess (the pytest parent must not touch CUDA before forking the
vLLM engine). See tests/test_dmillm_e2e.py.

Usage:
    python -m tests.vllm_dmillm_runner --result-file /tmp/r.json
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import uuid

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


def _shapes(outputs):
    return [(len(o.dmi_internal.hidden_states), o.dmi_internal.hidden_states[0].shape[1])
            for o in outputs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--result-file", required=True)
    args = ap.parse_args()

    import torch
    import clickhouse_driver
    from vllm import SamplingParams
    from dmi_vllm_integration.llm import DMILLM

    host = os.environ.get("DMX_DB_HOST", "localhost")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    database = os.environ.get("DMX_DB_DATABASE", "default")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database) is None:
        raise ValueError(f"invalid ClickHouse database: {database!r}")
    table = f"dmi_vllm_test_{uuid.uuid4().hex[:12]}"
    model_id = f"test_dmillm::{os.getpid()}"
    client = clickhouse_driver.Client(host=host, port=port)
    atexit.register(
        client.execute,
        f"DROP TABLE IF EXISTS {database}.{table} SYNC",
    )

    llm = DMILLM(
        args.model,
        additional_config={
            "dmx_model_id": model_id, "dmx_hook_selection": "resid_pre",
            "dmx_db_host": host, "dmx_db_port": port,
            "dmx_db_database": database, "dmx_db_table": table,
            "dmx_drain_flush_timeout_us": 100_000,
        },
        max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.5,
    )
    prompts = ["The capital of France is", "Hello"]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))

    expected_layers = int(
        llm.llm_engine.model_config.hf_config.num_hidden_layers
    )
    for output in outputs:
        output.dmi_internal.require(
            "hidden_states",
            count=expected_layers,
            retry=True,
            timeout_s=30.0,
            poll_s=0.25,
        )
    no_stop = _shapes(outputs)

    # Terminal drain clears each handle's live-read cache before this
    # authoritative re-read.
    llm.stop_monitoring()
    internals = [o.dmi_internal.hidden_states for o in outputs]
    after_stop = [(len(hs), hs[0].shape[1]) for hs in internals]

    tests = [
        {"name": "outputs_are_native",
         "passed": len(outputs) == 2 and all(o.outputs[0].text for o in outputs),
         "detail": [o.outputs[0].text for o in outputs]},
        # A live read may observe an asynchronously persisted prefix.  The
        # terminal stop clears that cache and makes the final read authoritative.
        {"name": "live_read_then_authoritative_stop",
         "passed": (
             all(layers == expected_layers for layers, _ in no_stop)
             and all(layers == expected_layers for layers, _ in after_stop)
             and all(final_seq >= live_seq > 0
                     for (_, live_seq), (_, final_seq) in zip(no_stop, after_stop))
         ),
         "detail": {"no_stop": no_stop, "after_stop": after_stop}},
        # Per-request: each is its own [1, seq, hidden] (batch dim 1).
        {"name": "per_request_hidden_states",
         "passed": all(len(hs) > 0 and hs[0].dim() == 3 and hs[0].shape[0] == 1
                       for hs in internals),
         "detail": after_stop},
        # Ragged prompts -> independent seq lengths (no cross-padding).
        {"name": "per_request_isolated_lengths",
         "passed": len({s for _, s in after_stop}) > 1,
         "detail": [s for _, s in after_stop]},
        {"name": "available_lists_hidden_states",
         "passed": "hidden_states" in outputs[0].dmi_internal.available,
         "detail": outputs[0].dmi_internal.available},
    ]

    with open(args.result_file, "w") as f:
        json.dump({"tests": tests}, f)

    del llm
    torch.cuda.empty_cache()
    sys.exit(0 if all(t["passed"] for t in tests) else 1)


if __name__ == "__main__":
    main()
