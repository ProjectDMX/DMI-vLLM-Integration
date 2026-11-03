"""Per-request and per-HookPoint capture selection for the vLLM adapter.

vLLM packs every scheduled request's rows into one activation tensor per
HookPoint.  By default DMI copies all of them.  This module lets a
deployment copy less, with the selection applied on the GPU by DMI's
request-row gather producer (``dmi.hook_rows``):

request mask
    ``set_request_mask(allowlist)`` or an in-process mask provider selects
    which requests are captured at all.  A request masked while in flight is
    *tainted*: its capture is incomplete, so its completion is reported
    ``valid=False``.
shared row table (``dmx_request_gather``)
    Every HookPoint copies only the masked requests' rows, compacted.
per-HookPoint row tables (``dmx_per_hook_request_gather``)
    Each HookPoint has its own table, so each request can select its own
    HookPoints and absolute token windows
    (``set_hook_request_selection``), or a capture planner supplies that
    demand every step.  With ``dmx_prefill_selection`` the tables' disabled
    marker also gates eager and piecewise-graph producers, so the node toggle
    follows the planned demand on every step.
in-graph heads (``dmx_inline_head_slots``)
    Fused projection heads score the selected ``resid_pre`` / ``final_ln``
    rows inside the graph; ``inline_register`` writes weights between steps
    without recapture.

A *capture planner* is an object offered by the sink (its
``capture_planner`` attribute) with:

``capture_plan(req_ids, token_ranges)``
    Called each step with the step's request IDs and absolute token ranges.
    Returns demand ``{rows, windows}`` (as attributes, optionally wrapped in
    ``.demands``) and optionally a ``capture_lease``
    (``dmi.transport.native.CaptureLease``) that rides on every row the
    step captures.  ``rows`` maps request IDs to sets of ``(hook_type,
    layer)`` pairs; ``windows`` maps request IDs to per-pair absolute
    half-open token intervals.  Return the same demand object while the
    demand is unchanged: the planner reuses compiled rules by identity.
``capture_plan_with_layout(req_ids, token_ranges, row_offsets, total_rows)``
    Optional variant that also receives the packed layout.
``admit_capture_inventory(inventory)``
    Called once after attachment with ``{(act_name, layer): (hook_type,
    layer)}`` for every HookPoint whose rows can be selected.
``bind_capture_transport(row_bytes, ring_engine, max_segments)``
    Optional; receives each HookPoint's row stride and the ring engine.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

HookKey = Tuple[int, int]

CAPTURE_MODES = ("dynamic", "static_full")
HEAD_WEIGHT_DTYPES = ("fp32", "bf16")

# Mirrors of native/csrc/ring/producer.cuh (the fused head node's arena).
HEAD_MAX_SLOTS = 16
HEAD_HDR_WORDS = 64
HEAD_BIAS_WORD = 32
HEAD_MAX_ACC = 16
HEAD_MODES = {"single": 1, "first": 2, "mid": 3, "last": 4}


def _hook_rows():
    try:
        from dmi import hook_rows
    except ImportError as exc:
        raise RuntimeError(
            "request-row gather requires DMI with per-HookPoint row tables "
            "(dmi.hook_rows)") from exc
    return hook_rows


def _head_types():
    from dmi.hooks.specs import HEAD_SOURCE_TYPE, HEAD_TYPES
    return HEAD_TYPES, HEAD_SOURCE_TYPE


def act_names() -> Dict[int, str]:
    """Hook type -> canonical activation name (``blocks.`` prefixed for
    per-layer hooks)."""
    from dmi.hooks.catalog import HOOK_DEFS
    return {d[0]: ("blocks." + d[1] if d[3] else d[1]) for d in HOOK_DEFS}


def with_inline_heads(specs: Sequence[Any], slots: int) -> List[Any]:
    """Insert one in-graph head spec after every token-row spec that has a
    head type (``resid_pre`` -> ``resid_pre_head``, ``final_ln`` ->
    ``final_ln_head``).

    The head spec names the same HookPoint: the fused node launches the
    heads right after the row producer, so metadata order equals launch
    order."""
    if not slots:
        return list(specs)
    from dmi.hooks.specs import HookSpec
    _, head_source = _head_types()
    sources = {src: head for head, src in head_source.items()}
    out = []
    for spec in specs:
        out.append(spec)
        head = sources.get(spec.hook_type)
        if head is not None and spec.dim0_is_actual_tokens:
            out.append(HookSpec(hook_type=head, module=spec.module,
                                layer_no=spec.layer_no, dtype=torch.float32,
                                dim0_is_actual_tokens=True))
    return out


class CaptureSelection:
    """Selection state of one ``VLLMAdaptor``; see the module docstring."""

    MASK_TAINT_CAP = 1 << 16

    def __init__(self, *, request_gather: bool = False,
                 per_hook_request_gather: bool = False,
                 prefill_selection: bool = False,
                 capture_mode: str = "dynamic",
                 inline_head_slots: int = 0,
                 inline_head_weight: str = "fp32",
                 table_span_upload: bool = True,
                 gather_host_skip: bool = True) -> None:
        if capture_mode not in CAPTURE_MODES:
            raise ValueError(f"capture mode must be one of {CAPTURE_MODES}")
        if inline_head_weight not in HEAD_WEIGHT_DTYPES:
            raise ValueError(f"inline head weights must be one of {HEAD_WEIGHT_DTYPES}")
        if type(inline_head_slots) is not int or not 0 <= inline_head_slots <= HEAD_MAX_SLOTS:
            raise ValueError(f"inline head slots must be an int in [0, {HEAD_MAX_SLOTS}]")
        per_hook = bool(per_hook_request_gather)
        if prefill_selection and not per_hook:
            raise ValueError("prefill selection requires per-HookPoint gather")
        if inline_head_slots and not per_hook:
            raise ValueError("in-graph heads require per-HookPoint gather")
        if capture_mode == "static_full" and (request_gather or per_hook):
            raise ValueError("static-full capture does not select rows")
        self.request_gather = bool(request_gather) or per_hook
        self.per_hook = per_hook
        self.prefill_selection = bool(prefill_selection)
        self.capture_mode = capture_mode
        self.head_slots = inline_head_slots
        self.head_bf16 = inline_head_weight == "bf16"
        self.table_span_upload = bool(table_span_upload)
        self.gather_host_skip = bool(gather_host_skip)

        # Request mask.
        self._allowlist: Optional[frozenset] = None
        self._capture_new = True
        self._snapshot: frozenset = frozenset()
        self._provider = None
        self._provider_failed = False
        self._tainted: set = set()
        self._taint_warned = False
        self._last_step_req_ids: List[str] = []
        self._counts = dict(steps_filtered=0, requests_masked=0)

        # Row tables.
        self.wired = False
        self.keys: Tuple[HookKey, ...] = ()
        self.max_segments = 0
        self._ring_engine = None
        self._transport = None
        self._table_dev: Optional[torch.Tensor] = None
        self._table_host: Optional[torch.Tensor] = None
        self._copy_event = None
        self._copy_pending = False
        self._uploaded: Optional[tuple] = None
        self._step_rows: Optional[dict] = None
        self._step_segments: Optional[list] = None
        self._step_abs_offsets: Optional[list] = None
        self._step_total_rows = 0
        self._step_gated = True
        self._selection = None
        self._packed = None
        self._provided_demands = None
        self._planned_enabled: Optional[frozenset] = None
        self.row_counts = dict(
            gather_steps_identity=0, gather_steps_partial=0,
            gather_steps_capacity_fallback=0, gather_rows_selected=0,
            gather_rows_full=0, hook_rows_selected=0, hook_rows_rectangular=0,
            hook_capacity_fallbacks=0, hook_plan_group_builds=0,
            hook_plan_group_reuses=0, planned_steps=0, gated_steps=0,
            enabled_hooks=0, enabled_zero_row_hooks=0)
        self.table_counts = dict(
            hook_table_uploads=0, hook_table_reuses=0, hook_table_upload_bytes=0,
            hook_table_wait_calls=0, hook_table_wait_ns=0, hook_table_wait_errors=0)
        self._warned_capacity = False

        # Capture planner.
        self.planner = None
        self._layout_planner = None

        # In-graph heads.
        self._head_keys: Tuple[HookKey, ...] = ()
        self._head_arena: Optional[torch.Tensor] = None
        self._head_acc: Optional[torch.Tensor] = None
        self._head_meta: List[list] = []
        self._head_live: Dict[HookKey, set] = {}
        self._head_weight_words = 0
        self._hidden = 0
        self.head_counts = dict(registers=0, retires=0, rows_scored=0,
                                scores=0, rows_exported=0)

    # ------------------------------------------------------------------
    # Attachment

    @property
    def active(self) -> bool:
        """Whether this adapter does anything beyond full capture."""
        return (self.request_gather or self.planner is not None
                or self._provider is not None or self._allowlist is not None)

    def install_planner(self, planner: Any) -> None:
        """Bind a capture planner before attachment."""
        if planner is None:
            return
        if self.capture_mode == "dynamic" and not (self.per_hook and self.prefill_selection):
            raise ValueError(
                "a capture planner requires per-HookPoint gather and prefill "
                "selection (or static-full capture)")
        plan = getattr(planner, "capture_plan", None)
        if not callable(plan) or not callable(getattr(planner, "admit_capture_inventory", None)):
            raise TypeError("a capture planner needs capture_plan and admit_capture_inventory")
        self.planner = planner
        self._layout_planner = getattr(planner, "capture_plan_with_layout", None)

    def wire(self, *, specs: Sequence[Any], cfg: Any, device: Any,
             ring_engine: Any, transport: Any, max_num_seqs: int,
             max_num_batched_tokens: int, row_bytes,
             max_capture_rows: int = 0) -> None:
        """Give every gather-capable HookPoint its row table; all or none.

        On a step with a shared table the request offsets are compacted for
        every flat-sliced hook at once, so a hook that cannot gather would
        be mis-sliced: such a configuration is refused (per-HookPoint) or
        falls back to full capture with a warning (shared table).
        ``final_logits`` is position-indexed (``orig_positions``) and is
        exempt; attention-weight matrices are not row-sliced."""
        self._ring_engine = ring_engine
        self._transport = transport
        self._hidden = int(cfg.hidden_dim)
        if not self.request_gather:
            return
        from dmi.hooks.specs import HOOK_TYPE_FINAL_LOGITS, _ATTN_WT_TYPES
        hook_rows = _hook_rows()
        head_types, _ = _head_types()
        eligible, blockers = [], []
        for spec in specs:
            if spec.hook_type == HOOK_TYPE_FINAL_LOGITS or spec.hook_type in _ATTN_WT_TYPES:
                continue
            rb = row_bytes(spec)
            if not spec.dim0_is_actual_tokens or rb <= 0:
                blockers.append((spec.hook_type, spec.layer_no))
                continue
            eligible.append((spec, rb))
        if blockers:
            if self.per_hook:
                raise ValueError(f"per-HookPoint gather requires token-row hooks: {blockers}")
            warnings.warn(
                f"[vllm_integration] request gather disabled: {len(blockers)} "
                f"active hook(s) cannot gather (not token rows): {blockers[:4]}",
                stacklevel=2)
            return
        if self.prefill_selection and len(eligible) != len(specs):
            raise ValueError("prefill selection requires a row table at every HookPoint")
        max_segs = min(int(max_num_seqs or 0) or hook_rows.GATHER_MAX_SEGS,
                       hook_rows.GATHER_MAX_SEGS)
        cuda = torch.device(device).type == "cuda"
        self.max_segments = max_segs
        width = 1 + 2 * max_segs
        if not eligible:
            # A TP/PP rank can hold no local token-row hook; planning still
            # runs (the planner tracks requests), with no table.
            self.keys = ()
            self.wired = True
            transport.set_prefill_table_gate(self.prefill_selection)
            return
        if not self.per_hook:
            # One shared table; zeros (n = 0) until the first step writes it.
            self._table_dev = torch.zeros(width, dtype=torch.int64, device=device)
            self._table_host = torch.zeros(width, dtype=torch.int64, pin_memory=cuda)
            self._copy_event = torch.cuda.Event() if cuda else None
            for spec, rb in eligible:
                hook_rows.configure_hook_gather(spec.module, self._table_dev, rb)
            self.keys = tuple((s.hook_type, s.layer_no) for s, _ in eligible)
            self.wired = True
            return

        keys = tuple((s.hook_type, s.layer_no) for s, _ in eligible)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate HookPoints in the per-HookPoint gather inventory")
        # One allocation: each HookPoint's producer bakes a fixed row view.
        self._table_dev = torch.zeros((len(keys), width), dtype=torch.int64, device=device)
        self._table_host = torch.zeros((len(keys), width), dtype=torch.int64, pin_memory=cuda)
        self._copy_event = torch.cuda.Event() if cuda else None
        heads = [(i, spec) for i, (spec, _) in enumerate(eligible) if spec.hook_type in head_types]
        if heads:
            K, hidden = self.head_slots, self._hidden
            w = (K * hidden + 1) // 2 if self.head_bf16 else K * hidden
            self._head_weight_words = w
            words = (HEAD_HDR_WORDS + w + 3) & ~3
            self._head_arena = torch.zeros((len(heads), words), dtype=torch.int32, device=device)
            nan = torch.full((HEAD_MAX_SLOTS,), float("nan"), dtype=torch.float32)
            self._head_arena[:, HEAD_BIAS_WORD:HEAD_BIAS_WORD + HEAD_MAX_SLOTS] = \
                nan.view(torch.int32).to(device)
            self._head_meta = [[(0, 0, float("nan"))] * HEAD_MAX_SLOTS for _ in heads]
            # Multi-layer partial sums, indexed by the row's position in one
            # forward's token batch (padded up to a captured graph size).
            rows = max(int(max_num_batched_tokens or 0), int(max_capture_rows or 0), 1)
            self._head_acc = torch.zeros((rows, HEAD_MAX_ACC), dtype=torch.float32, device=device)
            self._head_keys = tuple((s.hook_type, s.layer_no) for _, s in heads)
            self._head_live = {key: set() for key in self._head_keys}
        head_row = {i: j for j, (i, _) in enumerate(heads)}
        for i, (spec, rb) in enumerate(eligible):
            if spec.hook_type in head_types:
                hook_rows.configure_hook_heads(
                    spec.module, head_type=spec.hook_type, layer_no=spec.layer_no,
                    table=self._table_dev[i], arena=self._head_arena[head_row[i]],
                    acc=self._head_acc, slots=self.head_slots,
                    weight_bf16=self.head_bf16)
            else:
                hook_rows.configure_hook_gather(spec.module, self._table_dev[i], rb)
        if self.gather_host_skip:
            # Eager launches of a disabled HookPoint are skipped on the host
            # from the pinned row; graphs always launch and read the device row.
            ring_engine.set_gather_host_mirror(
                self._table_dev.data_ptr(), self._table_host.data_ptr(),
                self._table_dev.stride(0) * self._table_dev.element_size(), len(keys))
        self.keys = keys
        self.wired = True
        # Only now does every producer read its own fixed-address table.
        transport.set_prefill_table_gate(self.prefill_selection)

    def admit(self, specs: Sequence[Any], cfg: Any, row_bytes) -> None:
        """Hand the planner the selectable inventory (after ``wire``)."""
        planner = self.planner
        if planner is None:
            return
        static = self.capture_mode == "static_full"
        if not static and not (self.per_hook and self.prefill_selection and self.wired):
            raise ValueError("a capture planner requires per-HookPoint gather and prefill selection")
        names = act_names()
        keys = self.keys
        if static:
            if any(not s.dim0_is_actual_tokens or row_bytes(s) <= 0 for s in specs):
                raise ValueError("static-full capture requires token-row HookPoints")
            keys = tuple((s.hook_type, s.layer_no) for s in specs)
        planner.admit_capture_inventory(
            {(names[kind], layer): (kind, layer) for kind, layer in keys})
        bind = getattr(planner, "bind_capture_transport", None)
        if bind is not None and not static:
            strides = {(names[s.hook_type], s.layer_no): row_bytes(s)
                       for s in specs if (s.hook_type, s.layer_no) in set(self.keys)}
            bind(strides, self._ring_engine, self.max_segments)

    # ------------------------------------------------------------------
    # Request mask

    def set_request_mask(self, allowlist: Optional[Iterable[str]], *,
                         capture_new: bool = True, normalize=str) -> dict:
        """``allowlist=None`` captures every request (zero-cost fast path).
        Otherwise only the listed requests -- plus, with ``capture_new``,
        requests first scheduled after this call -- are captured from the
        next step.  In-flight requests not listed are masked (and tainted)."""
        if self.capture_mode == "static_full" and allowlist is not None:
            raise ValueError("static-full capture cannot filter requests")
        self._allowlist = (None if allowlist is None
                           else frozenset(normalize(str(r)) for r in allowlist))
        self._capture_new = bool(capture_new)
        self._snapshot = frozenset(self._last_step_req_ids)
        return self.mask_state()

    def set_request_mask_provider(self, provider: Any) -> None:
        """``provider(req_ids, token_ranges)`` returns the captured IDs each
        step (or None for all).  It takes precedence over the allowlist.  A
        raising provider fails open (captures every request) for that step,
        warns once, and is retried next step."""
        if self.capture_mode == "static_full" and provider is not None:
            raise ValueError("static-full capture cannot install a request mask provider")
        self._provider = provider
        self._provider_failed = False

    def is_tainted(self, req_id: str) -> bool:
        """True when the request's capture missed at least one step."""
        return req_id in self._tainted

    def clear_taint(self, req_id: str) -> None:
        self._tainted.discard(req_id)

    def _captured(self, req_ids, token_ranges) -> Optional[set]:
        provider = self._provider
        if provider is not None:
            try:
                chosen = provider(list(req_ids), list(token_ranges))
            except Exception as exc:
                if not self._provider_failed:
                    self._provider_failed = True
                    warnings.warn(
                        f"[vllm_integration] request-mask provider raised ({exc!r}); "
                        "capturing every request until it succeeds", stacklevel=3)
                return None
            return None if chosen is None else set(chosen)
        allow = self._allowlist
        if allow is None:
            return None
        if self._capture_new:
            snapshot = self._snapshot
            return {r for r in req_ids if r in allow or r not in snapshot}
        return {r for r in req_ids if r in allow}

    # ------------------------------------------------------------------
    # Per-HookPoint selection

    def set_hook_request_selection(self, rows, default=None, token_windows=None, *,
                                   normalize=str) -> dict:
        """Replace request -> HookPoint demand from the next step.

        ``rows`` maps request IDs to ``(hook_type, layer)`` pairs; an explicit
        empty entry selects nothing, an absent request uses ``default``, and
        ``default=None`` selects every gather-capable HookPoint.
        ``token_windows={request: {(hook_type, layer): [(start, end)]}}``
        restricts selected pairs to absolute half-open positions (``end=None``
        is unbounded); a missing entry means every position, ``[]`` none."""
        if self.planner is not None:
            raise RuntimeError("a capture planner owns the selection")
        if not self.per_hook or not self.wired:
            raise RuntimeError("per-HookPoint gather must be enabled before graph capture")
        hook_rows = _hook_rows()
        normalized, windows = {}, {}
        for rid, hooks in rows.items():
            key = normalize(str(rid))
            if key in normalized:
                raise ValueError(f"request IDs collide after normalization: {key!r}")
            normalized[key] = self._validate_pairs(hooks)
        for rid, by_hook in (token_windows or {}).items():
            key = normalize(str(rid))
            if key in windows:
                raise ValueError(f"window request IDs collide after normalization: {key!r}")
            windows[key] = {next(iter(self._validate_pairs([hook]))):
                            hook_rows.canonical_token_windows(spans)
                            for hook, spans in by_hook.items()}
        self._selection = hook_rows.HookRowPlanner(
            self.keys, normalized, self._validate_pairs(default),
            hook_rows.compile_token_windows(windows) if windows else None)
        return {"requests": len(normalized), "hooks": len(self.keys), "per_hook": True}

    def _validate_pairs(self, hooks):
        if hooks is None:
            return None
        parsed = []
        for pair in hooks:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) is not int for v in pair)):
                raise ValueError("HookPoint keys must be integer (hook_type, layer) pairs")
            parsed.append(tuple(pair))
        result = frozenset(parsed)
        unknown = result - frozenset(self.keys)
        if unknown:
            raise ValueError(f"HookPoints not wired for gather: {sorted(unknown)}")
        return result

    # ------------------------------------------------------------------
    # Per step

    def plan_step(self, *, req_ids: List[str], token_ranges: List[Tuple[int, int]],
                  dim0_offsets: List[int], total_rows: int,
                  actual_q_len: Optional[int], gated: bool) -> Dict[str, Any]:
        """Apply this step's selection; returns the ``StepContext`` fields
        to replace.  Call after the step's gate (``set_gated_step(gated)``)
        is set."""
        self._last_step_req_ids = list(req_ids)
        self._step_total_rows = total_rows
        self._step_rows = None
        self._step_segments = None
        self._step_abs_offsets = None
        self._step_gated = bool(gated)
        changes: Dict[str, Any] = {}
        positions = None
        selected = self._captured(req_ids, token_ranges)
        if selected is not None:
            keep = [i for i, rid in enumerate(req_ids) if rid in selected]
            if len(keep) != len(req_ids):
                kept = set(keep)
                dropped = [rid for i, rid in enumerate(req_ids) if i not in kept]
                self._tainted.update(dropped)
                if len(self._tainted) > self.MASK_TAINT_CAP and not self._taint_warned:
                    self._taint_warned = True
                    warnings.warn(
                        f"[vllm_integration] more than {self.MASK_TAINT_CAP} masked "
                        "requests are awaiting completion; is anything finalizing them?",
                        stacklevel=2)
                self._counts["steps_filtered"] += 1
                self._counts["requests_masked"] += len(dropped)
                req_ids = [req_ids[i] for i in keep]
                token_ranges = [token_ranges[i] for i in keep]
                dim0_offsets = [dim0_offsets[i] for i in keep]
                positions = keep
                changes.update(req_ids=req_ids, token_ranges=token_ranges,
                               dim0_offsets=dim0_offsets, orig_positions=positions)
                if self.wired and not self.per_hook:
                    changes.update(self._compact(token_ranges, dim0_offsets))
        if self.wired and not self.per_hook:
            c = self.row_counts
            if self._step_segments is not None:
                actual_q_len = sum(n for _, n in self._step_segments)
                c["gather_steps_partial"] += 1
            else:
                actual_q_len = total_rows
                c["gather_steps_identity"] += 1
            c["gather_rows_selected"] += actual_q_len
            c["gather_rows_full"] += total_rows
            changes["actual_q_len"] = actual_q_len
        if self.per_hook and self.wired:
            changes["hook_rows"] = self._plan_hook_rows(
                req_ids, token_ranges, dim0_offsets,
                positions if positions is not None else range(len(req_ids)),
                total_rows, changes)
        elif self.capture_mode == "static_full":
            _, lease = self._snapshot_planner(req_ids, token_ranges, dim0_offsets, total_rows)
            if lease is None:
                raise RuntimeError("static-full capture requires a capture lease every step")
            changes["capture_lease"] = lease
        return changes

    def _compact(self, token_ranges, dim0_offsets) -> dict:
        """Shared-table gather: merged source runs and packed offsets."""
        lengths = [b - a for a, b in token_ranges]
        segs: list = []
        for off, n in zip(dim0_offsets, lengths):
            if n <= 0:
                continue
            if segs and segs[-1][0] + segs[-1][1] == off:
                segs[-1][1] += n
            else:
                segs.append([off, n])
        if len(segs) > self.max_segments:
            # Over table capacity: capture the full step (offsets stay absolute).
            self.row_counts["gather_steps_capacity_fallback"] += 1
            if not self._warned_capacity:
                self._warned_capacity = True
                warnings.warn(
                    f"[vllm_integration] request gather: {len(segs)} segments exceed "
                    f"the row table ({self.max_segments}); capturing such steps in full",
                    stacklevel=3)
            return {}
        self._step_segments = segs
        self._step_abs_offsets = list(dim0_offsets)
        packed, acc = [], 0
        for n in lengths:
            packed.append(acc)
            acc += n
        return {"dim0_offsets": packed}

    def _snapshot_planner(self, req_ids, token_ranges, offsets, total_rows):
        planner = self.planner
        if planner is None:
            return None, None
        if self._layout_planner is not None:
            supplied = self._layout_planner(req_ids, token_ranges, offsets, total_rows)
        else:
            supplied = planner.capture_plan(req_ids, token_ranges)
        lease = getattr(supplied, "capture_lease", None)
        return getattr(supplied, "demands", supplied), lease

    def _plan_hook_rows(self, req_ids, token_ranges, offsets, positions, total_rows,
                        changes) -> dict:
        hook_rows = _hook_rows()
        positions = list(positions)
        if self.planner is not None:
            demands, lease = self._snapshot_planner(req_ids, token_ranges, offsets, total_rows)
            if lease is not None:
                changes["capture_lease"] = lease
            self._provided_demands = demands
            planner = self._packed
            if planner is None:
                planner = self._packed = hook_rows.PackedDemandPlanner(
                    self.keys, self._validate_pairs)
            builds = planner.group_builds
            plans = planner.plan(req_ids, token_ranges, offsets, positions, total_rows,
                                 self.max_segments, demands.rows, demands.windows,
                                 frozenset(), demand_identity=demands)
            rebuilt = planner.group_builds != builds
            present = planner.present_keys
        else:
            planner = self._selection
            if planner is None:
                planner = self._selection = hook_rows.HookRowPlanner(self.keys, {})
            builds = planner.group_builds
            plans = planner.plan(req_ids, token_ranges, offsets, positions, total_rows,
                                 self.max_segments)
            rebuilt = planner.group_builds != builds
            present = plans
        c = self.row_counts
        c["hook_plan_group_builds"] += int(rebuilt)
        c["hook_plan_group_reuses"] += int(not rebuilt)
        transport = self._transport
        if self.planner is not None:
            enabled = frozenset(k for k in present if plans[k].row_count or plans[k].req_ids)
            c["planned_steps"] += 1
            c["gated_steps"] += int(self._step_gated)
            c["enabled_hooks"] += len(enabled)
            c["enabled_zero_row_hooks"] += sum(1 for k in enabled if not plans[k].row_count)
            if enabled != self._planned_enabled:
                # Host accounting changes now; node state follows at the next
                # replay of each graph (lazy apply in the replay guard).
                if self.keys:
                    transport.set_active_hooks_lazy(sorted(enabled))
                self._planned_enabled = enabled
        head_types, _ = _head_types()
        firing = {(s.hook_type, s.layer_no) for s in transport.specs_for_step()}
        rectangular = sum(b - a for a, b in token_ranges)
        for key in firing.intersection(plans):
            n = plans[key].row_count
            if key[0] in head_types:
                self.head_counts["rows_scored"] += n
                self.head_counts["scores"] += n * len(self._head_live.get(key, ()))
            else:
                self.head_counts["rows_exported"] += n
            c["hook_rows_selected"] += n
            c["hook_rows_rectangular"] += rectangular
            c["hook_capacity_fallbacks"] += int(plans[key].capacity_fallback)
        self._step_rows = plans
        return plans

    def full_payload(self, ctx: Any, total_rows: int) -> Dict[str, Any]:
        """Fields for a step the eager safety net runs (every hook copies its
        full tensor): an identity shared table, or per-HookPoint plans that
        keep the consumer-side selection over un-compacted rows."""
        changes: Dict[str, Any] = {}
        if self.wired and not self.per_hook and self._step_segments is not None:
            self._step_segments = None
            changes["actual_q_len"] = total_rows
            changes["dim0_offsets"] = list(self._step_abs_offsets)
        if ctx.hook_rows is not None:
            self._step_rows = {key: rows.full_payload(total_rows)
                               for key, rows in ctx.hook_rows.items()}
            changes["hook_rows"] = self._step_rows
        return changes

    def upload(self) -> None:
        """Write this step's row tables and copy them to the device on the
        current stream (before the forward that reads them)."""
        if not self.wired or self._table_dev is None or self._step_total_rows <= 0:
            return
        if not self.per_hook:
            segs = self._step_segments
            words = ((1, 0, self._step_total_rows) if segs is None else
                     (len(segs), *(v for seg in segs for v in seg)))
            if words == self._uploaded:
                self.table_counts["hook_table_reuses"] += 1
                return
            self._wait_for_previous_copy()
            t = self._table_host
            t[:len(words)] = torch.tensor(words, dtype=torch.int64)
            self._table_dev[:len(words)].copy_(t[:len(words)], non_blocking=True)
            self._record_copy(words, len(words) * t.element_size())
            return
        if self._step_rows is None:
            return
        signatures = self._signatures()
        if signatures == self._uploaded:
            self.table_counts["hook_table_reuses"] += 1
            return
        self._wait_for_previous_copy()
        written = self._fill(signatures, self._uploaded)
        stride = self._table_host.stride(0) * self._table_host.element_size()
        if self.table_span_upload and written is not None:
            # Device rows outside the rewritten span already equal their
            # pinned rows, so one contiguous span copy is exact.
            if written:
                lo, hi = written[0], written[-1] + 1
                self._table_dev[lo:hi].copy_(self._table_host[lo:hi], non_blocking=True)
                self.table_counts["hook_table_upload_bytes"] += (hi - lo) * stride
        else:
            self._table_dev.copy_(self._table_host, non_blocking=True)
            self.table_counts["hook_table_upload_bytes"] += len(self.keys) * stride
        self._record_copy(signatures, 0)

    def _wait_for_previous_copy(self) -> None:
        """The pinned table may still be the source of the previous step's
        copy: wait for that small copy only, not for the forward."""
        if not (self._copy_pending and self._copy_event is not None):
            return
        started = time.perf_counter_ns()
        try:
            self._copy_event.synchronize()
        except BaseException:
            self.table_counts["hook_table_wait_errors"] += 1
            raise
        finally:
            self.table_counts["hook_table_wait_ns"] += time.perf_counter_ns() - started
            self.table_counts["hook_table_wait_calls"] += 1

    def _record_copy(self, uploaded, nbytes: int) -> None:
        if self._copy_event is not None:
            self._copy_event.record()
            self._copy_pending = True
        self._uploaded = uploaded
        self.table_counts["hook_table_uploads"] += 1
        self.table_counts["hook_table_upload_bytes"] += nbytes

    def _signatures(self) -> tuple:
        """One signature per table row; None for a HookPoint that must not
        fire this step.

        With the table gate every non-firing row is disabled.  With in-graph
        heads the rows of a disabled half of a fused node are disabled too:
        the fused node stays enabled while either of its keys is, and each
        half publishes unless its own row is -1."""
        row_signature = _hook_rows().row_signature
        plans = self._step_rows
        firing = None
        if self.prefill_selection or self.head_slots:
            firing = {(s.hook_type, s.layer_no) for s in self._transport.specs_for_step()}
        return tuple(row_signature(plans[key]) if firing is None or key in firing else None
                     for key in self.keys)

    def _fill(self, signatures, previous) -> Optional[List[int]]:
        """Pack changed rows into the pinned table; returns the rewritten
        row indices, or None when every row was rewritten."""
        segment_array = _hook_rows().segment_array
        partial = previous is not None and len(previous) == len(signatures)
        groups: Dict[Any, List[int]] = {}
        for i, sig in enumerate(signatures):
            if not partial or sig != previous[i]:
                groups.setdefault(sig, []).append(i)
        target = self._table_host.numpy()
        for sig, rows in groups.items():
            index = rows[0] if len(rows) == 1 else rows
            if sig is None:
                target[index, 0] = -1   # no task at all, not an empty task
                continue
            values = segment_array(self._step_rows[self.keys[rows[0]]])
            target[index, 0] = len(values)
            if len(values):
                target[index, 1:1 + 2 * len(values)] = np.asarray(values).reshape(-1)
        if not partial:
            return None
        return sorted(i for rows in groups.values() for i in rows)

    # ------------------------------------------------------------------
    # In-graph heads

    def inline_register(self, entries: Iterable[dict]) -> dict:
        """Write or clear head weight slots between steps (no recapture).

        Each entry: ``{"hook": [head_type or act_name, layer], "slot": k,
        "weight": <hidden fp32 values or bytes>, "bias": b, "mode":
        "single" | "first" | "mid" | "last", "acc": a}``; ``"retire": True``
        frees the slot.  A linear head over several layers registers one
        entry per participating node (``first`` ... ``last``) sharing one
        accumulator column.  The copies are synchronized before returning,
        so a selection published afterwards sees the weights."""
        if self._head_arena is None:
            raise RuntimeError("in-graph heads are not installed (dmx_inline_head_slots=0)")
        entries = list(entries)
        K, hidden = self.head_slots, self._hidden
        index = {key: j for j, key in enumerate(self._head_keys)}
        by_name = {name: kind for kind, name in act_names().items()}

        def key_of(hook):
            kind, layer = hook
            return (by_name[kind] if isinstance(kind, str) else int(kind), int(layer))

        for e in entries:
            key, slot = key_of(e["hook"]), int(e["slot"])
            if key not in index or not 0 <= slot < K:
                raise ValueError(f"unknown in-graph head slot {key}/{slot}")
            if not e.get("retire"):
                mode = e.get("mode", "single")
                if mode not in HEAD_MODES:
                    raise ValueError(f"unknown in-graph head mode {mode!r}")
                if mode != "single" and not 0 <= int(e.get("acc", -1)) < HEAD_MAX_ACC:
                    raise ValueError("a multi-layer head slot needs an accumulator column in [0, 16)")
        dtype = torch.bfloat16 if self.head_bf16 else torch.float32
        touched = set()
        for e in entries:
            key, slot = key_of(e["hook"]), int(e["slot"])
            j = index[key]
            row = self._head_arena[j]
            weights = row[HEAD_HDR_WORDS:HEAD_HDR_WORDS + self._head_weight_words] \
                .view(dtype)[:K * hidden]
            if e.get("retire"):
                weights[slot * hidden:(slot + 1) * hidden].zero_()
                self._head_meta[j][slot] = (0, 0, float("nan"))
                self._head_live[key].discard(slot)
                self.head_counts["retires"] += 1
            else:
                w = e["weight"]
                w = (torch.frombuffer(bytearray(w), dtype=torch.float32)
                     if isinstance(w, (bytes, bytearray)) else torch.as_tensor(w, dtype=torch.float32))
                if w.numel() != hidden:
                    raise ValueError(f"in-graph head weight must have {hidden} values")
                weights[slot * hidden:(slot + 1) * hidden].copy_(w.to(device=row.device, dtype=dtype))
                self._head_meta[j][slot] = (HEAD_MODES[e.get("mode", "single")],
                                            int(e.get("acc", 0)), float(e["bias"]))
                self._head_live[key].add(slot)
                self.head_counts["registers"] += 1
            touched.add(j)
        for j in touched:
            header = torch.zeros(HEAD_HDR_WORDS, dtype=torch.int32)
            for k, (mode, acc, _) in enumerate(self._head_meta[j]):
                header[k], header[HEAD_MAX_SLOTS + k] = mode, acc
            header[HEAD_BIAS_WORD:HEAD_BIAS_WORD + HEAD_MAX_SLOTS] = torch.tensor(
                [b for _, _, b in self._head_meta[j]], dtype=torch.float32).view(torch.int32)
            self._head_arena[j, :HEAD_HDR_WORDS].copy_(header.to(self._head_arena.device))
        if self._head_arena.is_cuda:
            torch.cuda.synchronize(self._head_arena.device)
        return {"registered": sum(len(v) for v in self._head_live.values()),
                "keys": len(self._head_keys),
                "multi_layer_slots": sum(1 for meta in self._head_meta
                                         for mode, _, _ in meta if mode >= 2)}

    # ------------------------------------------------------------------
    # Introspection

    def row_table_stats(self) -> dict:
        """Row-table counters (``dmi.sync_stats.gather_sync_stats``)."""
        return dict(self.table_counts)

    def inline_state(self) -> dict:
        launches = getattr(self._ring_engine, "head_launches", None)
        return {"slots": self.head_slots, "head_nodes": len(self._head_keys),
                "registered": sum(len(v) for v in self._head_live.values()),
                **self.head_counts,
                "arena_bytes": (0 if self._head_arena is None
                                else self._head_arena.numel() * self._head_arena.element_size()),
                "head_launches": int(launches()) if callable(launches) else None}

    def mask_state(self) -> dict:
        allow = self._allowlist
        host_skips = getattr(self._ring_engine, "gather_host_skips", None)
        return {
            "capture_mode": self.capture_mode,
            "allowlist": sorted(allow) if allow is not None else None,
            "capture_new": self._capture_new,
            "snapshot_size": len(self._snapshot),
            "provider_installed": self._provider is not None,
            "provider_failed": self._provider_failed,
            **self._counts,
            "tainted_inflight": len(self._tainted),
            "gather_wired": self.wired,
            "per_hook_gather": self.per_hook and self.wired,
            "prefill_selection": self.prefill_selection,
            "planner_installed": self.planner is not None,
            **self.row_counts,
            **self.table_counts,
            "gather_host_skips": int(host_skips()) if callable(host_skips) else None,
            "inline_heads": self.inline_state(),
        }


__all__ = [
    "CAPTURE_MODES",
    "CaptureSelection",
    "act_names",
    "with_inline_heads",
]
