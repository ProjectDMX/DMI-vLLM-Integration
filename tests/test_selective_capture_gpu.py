"""End-to-end selective capture on a real vLLM engine (in-process consumer).

Each mode runs ``tests.vllm_selective_runner`` in a subprocess (the parent
never touches CUDA): Qwen3-0.6B with FULL_AND_PIECEWISE CUDA graphs, a
``BatchQueue`` sink registered through ``register_sink_factory``, and checks
against an HF reference on CPU.

Environment variables:
  DMI_VLLM_TEST_MODEL       model (default "Qwen/Qwen3-0.6B")
  DMI_VLLM_TEST_GPU_MEMORY  gpu_memory_utilization (default 0.2)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.vllm, pytest.mark.e2e]

L1, L2 = 3, 7
ACT, HEAD = "blocks.hook_resid_pre", "blocks.hook_resid_pre_head"


def _run(mode, tmp_path):
    out = tmp_path / f"{mode}.json"
    model = os.environ.get("DMI_VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    result = subprocess.run(
        [sys.executable, "-m", "tests.vllm_selective_runner", mode, model, str(out)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        pytest.fail(f"runner failed:\n{result.stderr[-4000:]}")
    return json.loads(out.read_text())


def _captured(verdict):
    return {(rid, act, layer): positions for rid, act, layer, positions in verdict["captured"]}


def _forwarded(verdict, rid):
    # The last sampled token is never forwarded.
    return list(range(verdict["lengths"][rid] - 1))


@pytest.mark.parametrize("mode", ["gather", "returned"])
def test_per_hook_rows_follow_the_selection_through_graph_replays(mode, tmp_path):
    verdict = _run(mode, tmp_path)
    captured = _captured(verdict)
    assert captured == {("0", ACT, L1): _forwarded(verdict, "0"),
                        ("1", ACT, L1): _forwarded(verdict, "1"),
                        ("1", ACT, L2): [2, 3, 4]}
    # The table gate turns every other HookPoint off on prefill steps too.
    assert {tuple(s) for s in verdict["step_layer_sets"]} <= {(L1,), (L2,), (L1, L2)}
    assert verdict["finalized"] == [["0", True], ["1", True], ["2", True]]
    assert verdict["submit_exceptions"] == 0
    assert verdict["state"]["hook_table_uploads"] > 0
    assert verdict["min_cosine"] > 0.99


def test_node_toggle_gates_full_graph_steps(tmp_path):
    verdict = _run("toggle", tmp_path)
    captured = _captured(verdict)
    layer_sets = {tuple(s) for s in verdict["step_layer_sets"]}
    # Decode steps replay toggled full graphs (one layer); prefill steps run
    # piecewise graphs or eagerly and fire every installed HookPoint.
    assert (L1,) in layer_sets
    assert all(s == (L1,) or len(s) > 2 for s in layer_sets)
    for rid in ("0", "1", "2"):
        assert captured[(rid, ACT, L1)] == _forwarded(verdict, rid)
        assert captured[(rid, ACT, L2)] == list(range(verdict["prompts"][rid]))
    assert verdict["min_cosine"] > 0.99


def test_fused_heads_with_only_the_head_key_enabled(tmp_path):
    """Node toggling without the table gate: decode steps replay full graphs
    in which only the head half of the fused node is enabled, so they deliver
    scores and no rows; prefill steps fire every installed HookPoint."""
    verdict = _run("fused_toggle", tmp_path)
    captured = _captured(verdict)
    prompt = verdict["prompts"]["0"]
    assert captured == {("0", ACT, L1): list(range(prompt)),
                        ("0", HEAD, L1): _forwarded(verdict, "0")}
    assert verdict["head_rel_err"] < 1e-3
    assert verdict["submit_exceptions"] == 0
    assert verdict["finalized"] == [["0", True], ["1", True], ["2", True]]


def test_in_graph_heads_score_the_selected_rows(tmp_path):
    verdict = _run("heads", tmp_path)
    captured = _captured(verdict)
    assert captured == {("0", ACT, L1): _forwarded(verdict, "0"),
                        ("0", HEAD, L1): _forwarded(verdict, "0")}
    assert verdict["head_rel_err"] < 1e-3
    assert verdict["free_slot_nan"] is True
    assert verdict["finalized"] == [["0", True], ["1", True], ["2", True]]
