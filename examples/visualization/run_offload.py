"""Capture a small vLLM run for DMI's visualization notebook.

The script writes Qwen3-0.6B activations under ``model_id=demo_vllm``.
Re-running it removes only that model ID's existing rows first.

Usage:
    python examples/visualization/run_offload.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


# DMI supports vLLM's V1 model runner. This must be set before importing vLLM.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


MODEL_ID = "demo_vllm"
HF_MODEL = "Qwen/Qwen3-0.6B"
HOOK_SELECTION = "token_ids,resid_pre,final_logits"


def _read_prompt() -> str:
    return (Path(__file__).resolve().parent / "prompt.txt").read_text().strip()


def _wipe_my_rows(db_host: str, db_port: int, database: str, table: str) -> None:
    """Delete this example's prior rows, tolerating a fresh database."""
    import clickhouse_driver

    client = clickhouse_driver.Client(host=db_host, port=db_port)
    try:
        client.execute(
            f"ALTER TABLE {database}.{table} DELETE WHERE model_id = %(model_id)s",
            {"model_id": MODEL_ID},
            settings={"mutations_sync": 2},
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(
            marker in message
            for marker in ("doesn't exist", "unknown table", "could not find table")
        ):
            return
        print(
            f"[demo] WARNING: pre-insert wipe failed "
            f"({type(exc).__name__}: {exc}); continuing.",
            file=sys.stderr,
            flush=True,
        )
    finally:
        client.disconnect()


def main() -> None:
    # Do not probe CUDA in this process before vLLM starts its engine workers.
    from dmi_vllm_integration.llm import DMILLM
    from vllm import SamplingParams

    db_host = os.environ.get("DMX_DB_HOST", "localhost")
    db_port = int(os.environ.get("DMX_DB_PORT", "9000"))
    db_database = os.environ.get("DMX_DB_DATABASE", "default")
    db_table = os.environ.get("DMX_DB_TABLE", "offload")

    _wipe_my_rows(db_host, db_port, db_database, db_table)

    prompt = _read_prompt()
    print(f"[demo] Prompt: {prompt!r}", flush=True)

    llm = DMILLM(
        model=HF_MODEL,
        additional_config={
            "dmx_model_id": MODEL_ID,
            "dmx_hook_selection": HOOK_SELECTION,
            "dmx_db_host": db_host,
            "dmx_db_port": db_port,
            "dmx_db_database": db_database,
            "dmx_db_table": db_table,
        },
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )
    try:
        outputs = llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=8),
        )
        decoded = outputs[0].outputs[0].text
    finally:
        # Terminal drain: no more requests may be submitted afterward.
        llm.stop_monitoring()

    print(f"[demo] Output:  {decoded!r}", flush=True)
    print(f"[demo] model_id = {MODEL_ID}", flush=True)


if __name__ == "__main__":
    main()
