"""Per-hook perturbation gate for the vLLM integration models."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from tests.isolate_hook import (
    _COPY_LINE_RE,
    _patched_source,
    compare_model_path,
    isolated_hook,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_COMPARE_SOURCE = dedent(
    """
    class Foo:
        def forward(self, x):
            self._buf_q[:x.shape[0]].copy_(x)
            self._buf_k[:x.shape[0]].copy_(x)
            module._buf_attn_scores[:, :, :x.shape[2], :x.shape[3]].copy_(x)
            self._buf_resid_pre[:x.shape[0], :x.shape[1]].copy_(x)
            return x

        def allocate(self):
            self._buf_q = torch.empty(...)
    """
).strip("\n")


def test_patcher_keeps_only_selected_copy() -> None:
    patched, commented = _patched_source(SAMPLE_COMPARE_SOURCE, "q")
    assert sorted(set(commented)) == ["attn_scores", "k", "resid_pre"]
    assert "self._buf_q[:x.shape[0]].copy_(x)" in patched
    assert "# ISOLATE: self._buf_k" in patched
    assert "self._buf_q = torch.empty(...)" in patched


@pytest.mark.parametrize("model_key", ["gpt2", "qwen3", "llama"])
def test_oracle_patch_round_trip_is_byte_identical(model_key: str) -> None:
    source = compare_model_path("vllm", model_key)
    original = source.read_bytes()
    with isolated_hook("vllm", model_key, "q") as (_, commented):
        assert source.read_bytes() != original
        assert commented
        assert "q" not in commented
    assert source.read_bytes() == original


@pytest.mark.parametrize("model_key", ["gpt2", "qwen3", "llama"])
@pytest.mark.parametrize("hook", ["q", "resid_pre", "final_logits"])
def test_oracles_contain_representative_capture_points(
    model_key: str,
    hook: str,
) -> None:
    source = compare_model_path("vllm", model_key).read_text(encoding="utf-8")
    patched, _ = _patched_source(source, hook)
    retained = []
    for line in patched.splitlines():
        match = _COPY_LINE_RE.match(line.rstrip())
        if match is not None:
            retained.append(match.group("buf"))
    assert hook in retained


_RUNNER = dedent(
    """
    import argparse
    import os

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    import torch
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--hook", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout", choices=("orig", "ours", "ref"), required=True)
    args = parser.parse_args()

    models = {
        "gpt2": "gpt2",
        "qwen3": "Qwen/Qwen3-0.6B",
        "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    }
    kwargs = {
        "model": models[args.model_key],
        "max_model_len": 128,
        "gpu_memory_utilization": 0.5,
        "enforce_eager": args.mode == "eager",
    }
    if args.rollout == "ours":
        kwargs["worker_cls"] = "dmi_vllm_integration.worker.DMXGPUWorker"
        kwargs["additional_config"] = {
            "dmx_hook_selection": args.hook,
            "dmx_db_host": "",
        }
    elif args.rollout == "ref":
        kwargs["worker_cls"] = "tests.compare_worker.CompareWorker"
        kwargs["additional_config"] = {
            "dmx_hook_selection": args.hook,
            "dmx_db_host": "",
        }

    llm = LLM(**kwargs)
    outputs = llm.generate(
        ["Hello"], SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)
    )
    completion = outputs[0].outputs[0]
    ids = list(completion.token_ids)
    values = []
    for index, token_id in enumerate(ids):
        per_step = (completion.logprobs or [])[index]
        item = None if per_step is None else per_step.get(token_id)
        values.append(float("-inf") if item is None else item.logprob)
    torch.save(
        {
            "token_ids": torch.tensor(ids, dtype=torch.int64),
            "logprobs": torch.tensor(values, dtype=torch.float32),
        },
        os.path.join(args.output_dir, f"{args.rollout}.pt"),
    )
    if args.rollout != "orig":
        llm.collective_rpc("stop_monitoring")
    """
)


def _run_rollout(
    output_dir: Path,
    rollout: str,
    model_key: str,
    hook: str,
    mode: str,
) -> None:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            "--model-key",
            model_key,
            "--hook",
            hook,
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
            "--rollout",
            rollout,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert process.returncode == 0, (
        f"{rollout} failed\nstdout:\n{process.stdout}\nstderr:\n{process.stderr[-4000:]}"
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("hook", ["q", "resid_pre", "final_logits"])
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_qwen3_selected_hook_does_not_change_output(
    hook: str,
    mode: str,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    _run_rollout(tmp_path, "orig", "qwen3", hook, mode)
    _run_rollout(tmp_path, "ours", "qwen3", hook, mode)
    with isolated_hook("vllm", "qwen3", hook):
        _run_rollout(tmp_path, "ref", "qwen3", hook, mode)

    original = torch.load(tmp_path / "orig.pt", map_location="cpu")
    monitored = torch.load(tmp_path / "ours.pt", map_location="cpu")
    reference = torch.load(tmp_path / "ref.pt", map_location="cpu")
    assert torch.equal(original["token_ids"], monitored["token_ids"])
    assert torch.equal(original["token_ids"], reference["token_ids"])
    if mode == "eager":
        assert torch.equal(original["logprobs"], monitored["logprobs"])
        assert torch.equal(original["logprobs"], reference["logprobs"])
    else:
        assert torch.allclose(original["logprobs"], monitored["logprobs"], atol=0.15, rtol=0)
        assert torch.allclose(original["logprobs"], reference["logprobs"], atol=0.15, rtol=0)
