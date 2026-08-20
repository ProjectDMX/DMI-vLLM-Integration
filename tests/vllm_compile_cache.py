"""Cold-save/warm-load gate for vLLM's persistent AOT compile cache.

Run this module in an environment containing DMI, this integration package,
official vLLM 0.27.1, a cached GPT-2 checkpoint, and one visible GPU::

    python -m tests.vllm_compile_cache

The parent launches two fresh Python processes against one new cache root.  It
requires the cold process to save an AOT artifact, the warm process to load it
directly, and the serialized graph to retain DMI producer custom ops.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


_PASS_MARKER = "DMI_VLLM_COMPILE_CACHE_PASS"


def _child(model: str) -> None:
    import vllm
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            model=model,
            worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
            additional_config={
                "dmx_hook_selection": "vllm-full",
                "dmx_ring_payload_mb": 256,
                "dmx_ring_pinned_mb": 256,
                "dmx_db_host": "",
            },
            max_model_len=64,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            gpu_memory_utilization=0.35,
            enforce_eager=False,
        )
        outputs = llm.generate(
            ["Once upon a"],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs[0].token_ids) != 2:
            raise RuntimeError("compile-cache probe returned unexpected output")
        llm.collective_rpc("stop_monitoring")
    finally:
        if llm is not None:
            del llm

    print(f"{_PASS_MARKER} vllm={Path(vllm.__file__).resolve()}", flush=True)


def _run_child(
    *,
    model: str,
    cache_root: Path,
    inductor_root: Path,
    force_load: bool,
    log_path: Path,
) -> str:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_DISABLE_COMPILE_CACHE": "0",
            "VLLM_USE_AOT_COMPILE": "1",
            "VLLM_FORCE_AOT_LOAD": "1" if force_load else "0",
            "VLLM_CACHE_ROOT": str(cache_root),
            "TORCHINDUCTOR_CACHE_DIR": str(inductor_root),
            "VLLM_PLUGINS": "dmi_models",
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.vllm_compile_cache",
            "--child",
            "--model",
            model,
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"compile-cache child failed with exit {proc.returncode}; "
            f"see {log_path}"
        )
    if _PASS_MARKER not in proc.stdout:
        raise RuntimeError(f"compile-cache child omitted pass marker; see {log_path}")
    return proc.stdout


def _serialized_graphs(cache_root: Path) -> list[Path]:
    return sorted(cache_root.rglob("computation_graph.py"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        _child(args.model)
        return

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=False)
    else:
        artifact_dir = Path(tempfile.mkdtemp(prefix="dmi_vllm_compile_cache_"))

    cache_root = artifact_dir / "vllm-cache"
    inductor_root = artifact_dir / "inductor-cache"
    cold_log = artifact_dir / "cold-save.log"
    warm_log = artifact_dir / "warm-load.log"

    cold = _run_child(
        model=args.model,
        cache_root=cache_root,
        inductor_root=inductor_root,
        force_load=False,
        log_path=cold_log,
    )
    if "saved AOT compiled function" not in cold:
        raise RuntimeError(f"cold process did not save an AOT function; see {cold_log}")

    graphs = _serialized_graphs(cache_root)
    if not graphs:
        raise RuntimeError("cold process produced no serialized computation graph")
    if not any(
        "torch.ops.ring.producer" in path.read_text(encoding="utf-8")
        for path in graphs
    ):
        raise RuntimeError("serialized AOT graphs do not contain DMI producer ops")

    warm = _run_child(
        model=args.model,
        cache_root=cache_root,
        inductor_root=inductor_root,
        force_load=True,
        log_path=warm_log,
    )
    required_warm_markers = (
        "reconstructed serializable fn from standalone compile artifacts",
        "Directly load AOT compilation",
    )
    missing = [marker for marker in required_warm_markers if marker not in warm]
    if missing:
        raise RuntimeError(
            f"warm process did not prove direct AOT loading ({missing}); "
            f"see {warm_log}"
        )

    print(
        f"{_PASS_MARKER} cold-save warm-load artifacts={artifact_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
