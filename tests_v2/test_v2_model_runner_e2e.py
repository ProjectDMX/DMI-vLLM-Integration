"""Black-box output parity for stock and DMI-enabled vLLM V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def _run_inference(
    *,
    output: Path,
    raw_logits: Path,
    monitored: bool,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
            "E2E_MODEL": "qwen3",
            "E2E_DTYPE": "bfloat16",
            "E2E_ENFORCE_EAGER": "0",
            "E2E_MAX_NEW_TOKENS": "4",
            "E2E_MAX_MODEL_LEN": "128",
            "E2E_MAX_NUM_BATCHED_TOKENS": "128",
            "E2E_GPU_MEM_UTIL": "0.6",
            "E2E_RING_PAYLOAD_MB": "64",
            "E2E_RING_PINNED_MB": "64",
            "E2E_RAW_LOGITS_OUTPUT": str(raw_logits),
            "E2E_PROMPTS_JSON": json.dumps(
                [
                    "Paris is the capital of",
                    "Write one short sentence about deterministic testing.",
                    "Count upward from seven:",
                ]
            ),
            "DMX_HOOK_SELECTION": "resid_pre",
            "DMX_DB_HOST": "",
        }
    )
    command = [
        sys.executable,
        "-m",
        "tests_v2.vllm_logprob_runner",
        "--output",
        str(output),
    ]
    if monitored:
        command.append("--monitored")
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _assert_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        pytest.fail(
            f"{label} failed with rc={result.returncode}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    output = result.stdout + result.stderr
    assert "vllm.v1.worker.gpu.model_runner.GPUModelRunner" in output


def _assert_tensor_bits_equal(
    stock: torch.Tensor,
    monitored: torch.Tensor,
    label: str,
) -> None:
    assert stock.dtype is monitored.dtype, label
    assert stock.shape == monitored.shape, label
    assert torch.equal(
        stock.contiguous().view(torch.uint8),
        monitored.contiguous().view(torch.uint8),
    ), label


@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="CUDA not built"
)
@pytest.mark.gpu
@pytest.mark.vllm
@pytest.mark.e2e
def test_v2_stock_and_dmi_outputs_are_bitwise_identical(tmp_path: Path) -> None:
    stock_output = tmp_path / "stock-output.pt"
    monitored_output = tmp_path / "monitored-output.pt"
    stock_logits = tmp_path / "stock-raw-logits.pt"
    monitored_logits = tmp_path / "monitored-raw-logits.pt"

    stock_run = _run_inference(
        output=stock_output,
        raw_logits=stock_logits,
        monitored=False,
    )
    _assert_success(stock_run, "stock V2")
    monitored_run = _run_inference(
        output=monitored_output,
        raw_logits=monitored_logits,
        monitored=True,
    )
    _assert_success(monitored_run, "DMI V2")

    stock = torch.load(stock_output, weights_only=True, map_location="cpu")
    monitored = torch.load(
        monitored_output, weights_only=True, map_location="cpu"
    )
    assert stock.keys() == monitored.keys()
    for request_index in stock:
        stock_request = stock[request_index]
        monitored_request = monitored[request_index]
        for field in (
            "token_ids",
            "text",
            "finish_reason",
            "stop_reason",
            "prompt_token_ids",
        ):
            assert stock_request[field] == monitored_request[field], (
                request_index,
                field,
            )
        _assert_tensor_bits_equal(
            stock_request["logprobs"],
            monitored_request["logprobs"],
            f"request {request_index} public logprobs",
        )

    stock_raw = torch.load(stock_logits, weights_only=True, map_location="cpu")
    monitored_raw = torch.load(
        monitored_logits, weights_only=True, map_location="cpu"
    )
    assert len(stock_raw) == len(monitored_raw) > 0
    for call_index, (stock_call, monitored_call) in enumerate(
        zip(stock_raw, monitored_raw)
    ):
        _assert_tensor_bits_equal(
            stock_call,
            monitored_call,
            f"raw compute_logits call {call_index}",
        )
