"""Request and per-HookPoint capture selection (``capture_selection``).

Runs on CPU against DMI's Python planners: HookPoints, the transport and the
ring engine are fakes that record what the selection configures.  Native
delivery through the gather producer is covered by DMI's own GPU tests and
by ``test_selective_capture_gpu.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

hook_rows = pytest.importorskip("dmi.hook_rows")
if not hasattr(hook_rows, "configure_hook_gather"):
    pytest.skip("DMI without per-HookPoint row tables", allow_module_level=True)

from dmi.hooks.specs import (  # noqa: E402
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_RESID_PRE,
    HookSpec,
)

from dmi_vllm_integration.capture_selection import (  # noqa: E402
    CaptureSelection,
    act_names,
    with_inline_heads,
)

HID = 4
ROW_BYTES = HID * 2
R0, R1, M1 = (HOOK_TYPE_RESID_PRE, 0), (HOOK_TYPE_RESID_PRE, 1), (HOOK_TYPE_MLP_OUT, 1)


def _hook():
    return SimpleNamespace(_gather_table=None, _strip_row_bytes=0, _head_hook_type=None)


def _specs(keys=(R0, R1, M1)):
    return [HookSpec(hook_type=t, module=_hook(), layer_no=l, dim0_is_actual_tokens=True)
            for t, l in keys]


class Transport:
    def __init__(self, specs):
        self.specs = list(specs)
        self.enabled = None
        self.gated = True
        self.table_gate = False
        self.lazy_calls = []

    def set_prefill_table_gate(self, enabled):
        self.table_gate = enabled

    def set_active_hooks_lazy(self, enabled):
        self.lazy_calls.append(list(enabled))
        self.enabled = set(enabled)

    def specs_for_step(self):
        # As RingTransport: the enabled subset only on a gated step (or with
        # the table gate); an ungated step fires every installed spec.
        if self.enabled is None or not (self.gated or self.table_gate):
            return self.specs
        return [s for s in self.specs if (s.hook_type, s.layer_no) in self.enabled]


class RingEngine:
    def __init__(self):
        self.mirror = None

    def set_gather_host_mirror(self, *args):
        self.mirror = args

    def gather_host_skips(self):
        return 7


def _cfg():
    return SimpleNamespace(hidden_dim=HID, head_slots=0)


def _wired(specs=None, **options):
    specs = specs if specs is not None else _specs()
    selection = CaptureSelection(**options)
    transport, engine = Transport(specs), RingEngine()
    selection.wire(specs=specs, cfg=_cfg(), device="cpu", ring_engine=engine,
                   transport=transport, max_num_seqs=8, max_num_batched_tokens=64,
                   row_bytes=lambda spec: ROW_BYTES)
    return selection, transport, engine, specs


# A step: requests a (rows 0-1), b (rows 2-4), c (row 5).
STEP = dict(req_ids=["a", "b", "c"], token_ranges=[(5, 7), (0, 3), (9, 10)],
            dim0_offsets=[0, 2, 5], total_rows=6, actual_q_len=6, gated=True)


def _table(selection, key):
    row = selection._table_dev[selection.keys.index(key)].tolist()
    n = row[0]
    return n if n <= 0 else [tuple(row[1 + 2 * k:3 + 2 * k]) for k in range(n)]


# ---------------------------------------------------------------------------
# Configuration


@pytest.mark.parametrize(("options", "message"), [
    (dict(capture_mode="partial"), "capture mode"),
    (dict(inline_head_weight="fp16"), "inline head weights"),
    (dict(inline_head_slots=17, per_hook_request_gather=True), "inline head slots"),
    (dict(prefill_selection=True), "prefill selection requires"),
    (dict(inline_head_slots=2), "in-graph heads require"),
    (dict(capture_mode="static_full", request_gather=True), "static-full"),
])
def test_invalid_combinations_are_refused(options, message):
    with pytest.raises(ValueError, match=message):
        CaptureSelection(**options)


def test_default_selection_is_inactive_and_per_hook_implies_gather():
    assert CaptureSelection().active is False
    assert CaptureSelection(per_hook_request_gather=True).request_gather is True


def test_inline_heads_follow_their_source_specs():
    specs = _specs((R0, M1)) + [HookSpec(hook_type=HOOK_TYPE_FINAL_LN, module=_hook(),
                                         layer_no=-1, dim0_is_actual_tokens=True)]
    out = with_inline_heads(specs, 3)
    kinds = [(s.hook_type, s.layer_no) for s in out]
    assert kinds == [R0, (24, 0), M1, (HOOK_TYPE_FINAL_LN, -1), (25, -1)]
    assert out[1].module is out[0].module and out[1].dtype == torch.float32
    assert with_inline_heads(specs, 0) == specs
    assert act_names()[HOOK_TYPE_RESID_PRE] == "blocks.hook_resid_pre"


# ---------------------------------------------------------------------------
# Request mask


def test_mask_filters_requests_taints_the_masked_and_keeps_positions():
    selection = CaptureSelection()
    selection.plan_step(**STEP)                       # snapshot: a, b, c in flight
    selection.set_request_mask(["b"])
    changes = selection.plan_step(**dict(STEP, req_ids=["a", "b", "c", "d"],
                                         token_ranges=STEP["token_ranges"] + [(0, 1)],
                                         dim0_offsets=[0, 2, 5, 6], total_rows=7))
    # b is allowlisted and d arrived after the mask (capture_new).
    assert changes["req_ids"] == ["b", "d"]
    assert changes["orig_positions"] == [1, 3]
    assert changes["dim0_offsets"] == [2, 6]          # absolute without gather
    assert selection.is_tainted("a") and selection.is_tainted("c")
    assert not selection.is_tainted("d")
    selection.clear_taint("a")
    assert not selection.is_tainted("a")
    state = selection.mask_state()
    assert state["allowlist"] == ["b"] and state["requests_masked"] == 2


def test_strict_mask_and_the_all_sentinel():
    selection = CaptureSelection()
    selection.set_request_mask(["c"], capture_new=False)
    assert selection.plan_step(**STEP)["req_ids"] == ["c"]
    selection.set_request_mask(None)
    assert selection.plan_step(**STEP) == {}
    assert selection.active is False


def test_a_raising_provider_fails_open_and_is_retried():
    calls = []

    def provider(req_ids, ranges):
        calls.append(req_ids)
        if len(calls) == 1:
            raise RuntimeError("down")
        return {"a"}

    selection = CaptureSelection()
    selection.set_request_mask_provider(provider)
    with pytest.warns(UserWarning, match="request-mask provider raised"):
        assert selection.plan_step(**STEP) == {}
    assert selection.plan_step(**STEP)["req_ids"] == ["a"]
    assert selection.mask_state()["provider_failed"] is True


def test_static_full_refuses_request_filters():
    selection = CaptureSelection(capture_mode="static_full")
    with pytest.raises(ValueError, match="static-full"):
        selection.set_request_mask(["a"])
    with pytest.raises(ValueError, match="static-full"):
        selection.set_request_mask_provider(lambda *a: None)


# ---------------------------------------------------------------------------
# Shared row table


def test_shared_table_compacts_masked_rows_and_falls_back_on_eager():
    selection, transport, engine, specs = _wired(request_gather=True)
    assert all(s.module._gather_table is selection._table_dev for s in specs)
    assert all(s.module._strip_row_bytes == ROW_BYTES for s in specs)
    selection.set_request_mask(["a", "c"], capture_new=False)
    ctx_changes = selection.plan_step(**STEP)
    assert ctx_changes["dim0_offsets"] == [0, 2]        # packed destination rows
    assert ctx_changes["actual_q_len"] == 3
    selection.upload()
    assert selection._table_dev[:5].tolist() == [2, 0, 2, 5, 1]
    # The eager safety net copies full tensors: identity table, source offsets.
    ctx = SimpleNamespace(hook_rows=None)
    assert selection.full_payload(ctx, 6) == {"actual_q_len": 6, "dim0_offsets": [0, 5]}
    selection.upload()
    assert selection._table_dev[:3].tolist() == [1, 0, 6]


def test_shared_table_capacity_fallback_captures_the_step_in_full():
    selection, *_ = _wired(request_gather=True)
    selection.max_segments = 1
    selection.set_request_mask(["a", "c"], capture_new=False)
    with pytest.warns(UserWarning, match="exceed the row table"):
        changes = selection.plan_step(**STEP)
    assert changes["dim0_offsets"] == [0, 5] and changes["actual_q_len"] == 6
    assert selection.row_counts["gather_steps_capacity_fallback"] == 1


def test_a_hook_that_cannot_gather_disables_the_shared_table():
    specs = _specs((R0,)) + [HookSpec(hook_type=HOOK_TYPE_MLP_OUT, module=_hook(),
                                      layer_no=1, dim0_is_actual_tokens=False)]
    with pytest.warns(UserWarning, match="request gather disabled"):
        selection, *_ = _wired(specs, request_gather=True)
    assert selection.wired is False
    with pytest.raises(ValueError, match="token-row hooks"):
        _wired(specs, per_hook_request_gather=True)


def test_final_logits_is_position_indexed_and_needs_no_table():
    specs = _specs((R0,)) + [HookSpec(hook_type=HOOK_TYPE_FINAL_LOGITS, module=_hook(),
                                      layer_no=-1, dim0_is_actual_tokens=False)]
    selection, *_ = _wired(specs, request_gather=True)
    assert selection.keys == (R0,)
    assert specs[1].module._gather_table is None


# ---------------------------------------------------------------------------
# Per-HookPoint tables


def test_per_hook_tables_select_hooks_and_windows_per_request():
    selection, transport, engine, specs = _wired(per_hook_request_gather=True)
    assert engine.mirror is not None and engine.mirror[3] == 3
    assert all(s.module._gather_table.data_ptr() == selection._table_dev[i].data_ptr()
               for i, s in enumerate(specs))
    reply = selection.set_hook_request_selection(
        {"a": [list(R0)], "b": [R0, M1]}, default=[],
        token_windows={"b": {M1: [(1, None)]}})
    assert reply == {"requests": 2, "hooks": 3, "per_hook": True}
    plans = selection.plan_step(**STEP)["hook_rows"]
    assert plans[R0].req_ids == ("a", "b") and plans[R0].row_count == 5
    assert plans[M1].req_ids == ("b",) and plans[M1].token_ranges == ((1, 3),)
    assert plans[R1].row_count == 0
    selection.upload()
    assert _table(selection, R0) == [(0, 5)]
    assert _table(selection, M1) == [(3, 2)]
    assert _table(selection, R1) == 0                  # an empty task, not disabled
    assert selection.table_counts["hook_table_uploads"] == 1


def test_unchanged_tables_are_reused_and_changed_rows_span_upload():
    selection, *_ = _wired(per_hook_request_gather=True)
    selection.set_hook_request_selection({}, default=[R0])
    selection.plan_step(**STEP)
    selection.upload()
    selection.plan_step(**STEP)
    selection.upload()
    assert selection.table_counts["hook_table_reuses"] == 1
    # Only the last row changes: one row of the table is copied.
    stride = selection._table_host.stride(0) * 8
    before = selection.table_counts["hook_table_upload_bytes"]
    selection.set_hook_request_selection({}, default=[R0, M1])
    selection.plan_step(**STEP)
    selection._table_dev[0].fill_(99)                  # outside the span: untouched
    selection.upload()
    assert selection.table_counts["hook_table_upload_bytes"] - before == stride
    assert _table(selection, M1) == [(0, 6)]
    assert selection._table_dev[0, 0].item() == 99


def test_table_gate_disables_hooks_the_toggle_turned_off():
    selection, transport, *_ = _wired(per_hook_request_gather=True, prefill_selection=True)
    assert transport.table_gate is True
    transport.enabled = {R0}
    selection.plan_step(**STEP)
    selection.upload()
    assert _table(selection, R0) == [(0, 6)]
    assert _table(selection, R1) == -1 and _table(selection, M1) == -1


def test_selection_validation():
    selection, *_ = _wired(per_hook_request_gather=True)
    with pytest.raises(ValueError, match="integer"):
        selection.set_hook_request_selection({"a": [("0", 1)]})
    with pytest.raises(ValueError, match="not wired"):
        selection.set_hook_request_selection({"a": [(HOOK_TYPE_RESID_PRE, 9)]})
    with pytest.raises(ValueError, match="collide"):
        selection.set_hook_request_selection({"a-1": [], "a-2": []},
                                             normalize=lambda r: r.split("-")[0])
    unwired = CaptureSelection()
    with pytest.raises(RuntimeError, match="before graph capture"):
        unwired.set_hook_request_selection({})


def test_eager_fallback_keeps_per_hook_attribution_over_full_rows():
    selection, *_ = _wired(per_hook_request_gather=True)
    selection.set_hook_request_selection({"b": [R0]}, default=[])
    plans = selection.plan_step(**STEP)["hook_rows"]
    changes = selection.full_payload(SimpleNamespace(hook_rows=plans), 6)
    full = changes["hook_rows"][R0]
    assert full.segments == ((0, 6),) and full.row_count == 6
    assert full.req_ids == ("b",) and full.offsets == (2,)
    selection.upload()
    assert _table(selection, R0) == [(0, 6)]


# ---------------------------------------------------------------------------
# Capture planner


class Planner:
    def __init__(self, demands, lease=None):
        self.demands, self.lease = demands, lease
        self.inventory = self.strides = None
        self.calls = 0

    def admit_capture_inventory(self, inventory):
        self.inventory = inventory

    def bind_capture_transport(self, row_bytes, ring_engine, max_segments):
        self.strides = (row_bytes, max_segments)

    def capture_plan(self, req_ids, token_ranges):
        self.calls += 1
        return SimpleNamespace(demands=self.demands, capture_lease=self.lease)


def test_planner_demand_drives_rows_toggle_and_lease():
    demands = SimpleNamespace(rows={"a": frozenset([R0]), "c": frozenset([R0, M1])},
                              windows={})
    planner = Planner(demands, lease="lease-7")
    selection = CaptureSelection(per_hook_request_gather=True, prefill_selection=True)
    selection.install_planner(planner)
    specs = _specs()
    transport, engine = Transport(specs), RingEngine()
    selection.wire(specs=specs, cfg=_cfg(), device="cpu", ring_engine=engine,
                   transport=transport, max_num_seqs=8, max_num_batched_tokens=64,
                   row_bytes=lambda s: ROW_BYTES)
    selection.admit(specs, _cfg(), lambda s: ROW_BYTES)
    assert planner.inventory[("blocks.hook_resid_pre", 1)] == R1
    assert planner.strides[1] == 8
    changes = selection.plan_step(**STEP)
    assert changes["capture_lease"] == "lease-7"
    assert transport.lazy_calls == [sorted([R0, M1])]
    selection.plan_step(**STEP)                         # unchanged: no new toggle
    assert transport.lazy_calls == [sorted([R0, M1])]
    selection.upload()
    assert _table(selection, R0) == [(0, 2), (5, 1)]
    assert _table(selection, M1) == [(5, 1)]
    assert _table(selection, R1) == -1
    with pytest.raises(RuntimeError, match="planner owns"):
        selection.set_hook_request_selection({})


def test_a_planner_needs_table_gated_selection_or_static_full():
    with pytest.raises(ValueError, match="prefill selection"):
        CaptureSelection(per_hook_request_gather=True).install_planner(Planner(None))
    with pytest.raises(TypeError, match="capture_plan"):
        CaptureSelection(capture_mode="static_full").install_planner(object())


def test_static_full_passes_the_planner_lease_and_selects_nothing():
    planner = Planner(None, lease="epoch")
    selection = CaptureSelection(capture_mode="static_full")
    selection.install_planner(planner)
    assert selection.plan_step(**STEP) == {"capture_lease": "epoch"}
    planner.lease = None
    with pytest.raises(RuntimeError, match="lease"):
        selection.plan_step(**STEP)


# ---------------------------------------------------------------------------
# In-graph heads


def test_inline_register_writes_and_retires_arena_slots():
    specs = with_inline_heads(_specs((R0, R1)), 2)
    selection, transport, engine, _ = _wired(
        specs, per_hook_request_gather=True, inline_head_slots=2)
    head = specs[1].module
    assert head._head_hook_type == 24 and head._head_slots == 2
    arena = selection._head_arena
    assert arena.shape[0] == 2
    reply = selection.inline_register([
        {"hook": ["blocks.hook_resid_pre_head", 0], "slot": 1,
         "weight": [1.0, 2.0, 3.0, 4.0], "bias": 0.5},
        {"hook": [24, 1], "slot": 0, "weight": np.ones(HID, np.float32).tobytes(),
         "bias": 0.0, "mode": "first", "acc": 3},
    ])
    assert reply == {"registered": 2, "keys": 2, "multi_layer_slots": 1}
    header = arena[0, :64]
    assert header[1].item() == 1 and header[16 + 1].item() == 0
    assert header[32 + 1].view(torch.float32).item() == 0.5
    assert torch.isnan(header[32:48].view(torch.float32)[0])
    weights = arena[0, 64:64 + 2 * HID].view(torch.float32)
    assert weights[HID:2 * HID].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert arena[1, 16].item() == 3 and arena[1, 0].item() == 2
    selection.inline_register([{"hook": [24, 0], "slot": 1, "retire": True}])
    assert arena[0, 1].item() == 0 and weights[HID:].abs().sum().item() == 0
    with pytest.raises(ValueError, match="unknown in-graph head slot"):
        selection.inline_register([{"hook": [24, 0], "slot": 2, "weight": [0] * HID,
                                    "bias": 0}])
    with pytest.raises(ValueError, match="accumulator column"):
        selection.inline_register([{"hook": [24, 0], "slot": 0, "weight": [0] * HID,
                                    "bias": 0, "mode": "last"}])
    with pytest.raises(ValueError, match=f"{HID} values"):
        selection.inline_register([{"hook": [24, 0], "slot": 0, "weight": [0],
                                    "bias": 0}])
    assert selection.inline_state()["registers"] == 2
    assert CaptureSelection().inline_state()["head_nodes"] == 0
    with pytest.raises(RuntimeError, match="not installed"):
        CaptureSelection().inline_register([])


def test_bf16_head_weights_pack_two_per_word():
    specs = with_inline_heads(_specs((R0,)), 3)
    selection, *_ = _wired(specs, per_hook_request_gather=True, inline_head_slots=3,
                           inline_head_weight="bf16")
    assert selection._head_weight_words == (3 * HID + 1) // 2
    selection.inline_register([{"hook": [24, 0], "slot": 2, "weight": [0.5] * HID,
                                "bias": 1.0}])
    weights = selection._head_arena[0, 64:64 + selection._head_weight_words].view(torch.bfloat16)
    assert weights[2 * HID:3 * HID].float().tolist() == [0.5] * HID


def test_row_table_stats_and_state_report_counters():
    selection, *_ = _wired(per_hook_request_gather=True)
    selection.plan_step(**STEP)
    selection.upload()
    stats = selection.row_table_stats()
    assert stats["hook_table_uploads"] == 1 and stats["hook_table_wait_calls"] == 0
    state = selection.mask_state()
    assert state["per_hook_gather"] is True and state["gather_host_skips"] == 7
    assert state["hook_rows_selected"] == 18 and state["hook_rows_rectangular"] == 18


def test_a_disabled_half_of_a_fused_node_never_publishes():
    """Heads without the table gate: on a gated step with only the head key
    enabled, the fused node still runs, so the rows' table must be -1 or the
    node would publish a task nobody reserved."""
    specs = with_inline_heads(_specs((R0,)), 4)
    selection, transport, *_ = _wired(specs, per_hook_request_gather=True,
                                      inline_head_slots=4)
    assert transport.table_gate is False
    selection.set_hook_request_selection({}, default=[R0, (24, 0)])
    transport.enabled = {(24, 0)}
    selection.plan_step(**STEP)
    selection.upload()
    assert _table(selection, R0) == -1
    assert _table(selection, (24, 0)) == [(0, 6)]
    # An ungated step (eager or piecewise) fires every installed spec.
    transport.gated = False
    selection.plan_step(**dict(STEP, gated=False))
    selection.upload()
    assert _table(selection, R0) == [(0, 6)] and _table(selection, (24, 0)) == [(0, 6)]


class Event:
    def __init__(self):
        self.calls = []

    def record(self):
        self.calls.append("record")

    def synchronize(self):
        self.calls.append("synchronize")


def test_shared_table_waits_for_its_previous_copy_and_skips_unchanged_tables():
    selection, *_ = _wired(request_gather=True)
    selection._copy_event = event = Event()
    selection.set_request_mask(["a", "c"], capture_new=False)
    selection.plan_step(**STEP)
    selection.upload()
    assert event.calls == ["record"]
    selection.plan_step(**STEP)
    selection.upload()                                 # same table: no copy at all
    assert event.calls == ["record"]
    assert selection.table_counts["hook_table_reuses"] == 1
    selection.set_request_mask(["b"], capture_new=False)
    selection.plan_step(**STEP)
    selection.upload()                                 # rewrite only after the copy
    assert event.calls == ["record", "synchronize", "record"]
    assert selection._table_dev[:3].tolist() == [1, 2, 3]
    counts = selection.table_counts
    assert counts["hook_table_uploads"] == 2 and counts["hook_table_wait_calls"] == 1
    assert counts["hook_table_upload_bytes"] == (5 + 3) * 8


def test_head_accumulator_covers_padded_graph_rows():
    specs = with_inline_heads(_specs((R0,)), 2)
    selection = CaptureSelection(per_hook_request_gather=True, inline_head_slots=2)
    selection.wire(specs=specs, cfg=_cfg(), device="cpu", ring_engine=RingEngine(),
                   transport=Transport(specs), max_num_seqs=8, max_num_batched_tokens=64,
                   row_bytes=lambda spec: ROW_BYTES, max_capture_rows=512)
    assert selection._head_acc.shape == (512, 16)
