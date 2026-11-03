"""vLLM wiring for DMI's runtime CUDA-graph node toggling.

DMI (``dmi.node_toggle``) records each producer's kernel node while a
capture window is open and later enables or disables those nodes in the
instantiated graphs.  This module supplies the vLLM-specific parts:

* which captures enter the registry: only full-model graphs
  (``CUDAGraphMode.FULL``).  A piecewise segment graph holds only some
  layers' producers, which would make the registry non-uniform, so its
  capture is excluded; its producers run on every replay and are gated by
  the per-HookPoint row tables when those are installed;
* finding vLLM's captured full graphs after warmup and binding their execs;
* classifying each step: a step is gated exactly when vLLM dispatches it to
  a full graph.

Imported only when ``dmx_node_toggle`` is enabled, so DMI revisions without
node toggling still serve the default configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

_LOG = logging.getLogger(__name__)

HookKey = Tuple[int, int]


def _node_toggle():
    try:
        from dmi import node_toggle
    except ImportError as exc:
        raise RuntimeError(
            "dmx_node_toggle requires DMI with CUDA-graph node toggling "
            "(dmi.node_toggle)") from exc
    return node_toggle


def parse_enabled_hooks(value: Any) -> Optional[list]:
    """``dmx_enabled_hooks``: unset or ``""`` (gate inactive), ``"none"``
    (every hook off) or ``"hook_type:layer,..."``."""
    return _node_toggle().parse_enabled_hooks(value)


def capture_is_piecewise() -> bool:
    """True while vLLM captures a piecewise segment graph.

    Read from the forward context vLLM sets around every capture run; no
    context (a non-vLLM capture) counts as not piecewise."""
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
    except Exception:
        return False
    return getattr(ctx, "cudagraph_runtime_mode", None) == CUDAGraphMode.PIECEWISE


def step_is_gated(cudagraph_mode: Any) -> bool:
    """Whether a step dispatched with ``cudagraph_mode`` replays a toggled
    graph: only full graphs are registered and bound."""
    from vllm.config import CUDAGraphMode
    return cudagraph_mode == CUDAGraphMode.FULL


def begin_capture(engine: Any, *, device_toggle: bool = False) -> None:
    """Open the capture window before vLLM's warmup captures its graphs."""
    _node_toggle().begin_capture_window(
        engine, exclude_capture=capture_is_piecewise,
        device_toggle=device_toggle)


def full_graphs(model: Any) -> Iterable[Any]:
    """Every ``torch.cuda.CUDAGraph`` held by a full-mode
    ``CUDAGraphWrapper`` reachable from ``model``."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode

    seen, stack = set(), [model]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if (isinstance(obj, CUDAGraphWrapper)
                and obj.runtime_mode == CUDAGraphMode.FULL):
            for entry in obj.concrete_cudagraph_entries.values():
                graph = getattr(entry, "cudagraph", None)
                if graph is not None:
                    yield graph
        stack.append(getattr(obj, "runnable", None))


def finish_capture(model: Any, engine: Any, transport: Any,
                   enabled_hooks: Optional[Iterable[HookKey]], *,
                   lazy: bool) -> int:
    """Bind the captured full graphs, close the window and apply the static
    enabled set.  Call after warmup, before serving.  Returns the number of
    graphs bound."""
    toggle = _node_toggle()
    try:
        bound = 0
        for graph in full_graphs(model):
            toggle.bind_graph(engine, graph)
            bound += 1
    finally:
        toggle.end_capture_window(engine)
    _LOG.info("node toggle: bound %d full graph(s), %d producer node(s); "
              "%d piecewise capture(s) excluded", bound,
              engine.toggle_node_count(), toggle.excluded_capture_count())
    if enabled_hooks is not None and bound == 0:
        raise RuntimeError(
            "dmx_enabled_hooks requires a full CUDA graph (cudagraph_mode FULL, "
            "FULL_DECODE_ONLY or FULL_AND_PIECEWISE); none was captured, so "
            "the toggle would be silently inert.")
    if bound:
        # The guard validates every gated replay and, in lazy mode, applies
        # the deferred node state of the graph about to replay.
        toggle.set_replay_guard(True)
    toggle.activate(transport, enabled_hooks, lazy=lazy)
    return bound


def abort_capture(engine: Any) -> None:
    """Close the window after a failed warmup."""
    _node_toggle().end_capture_window(engine)


__all__ = [
    "abort_capture",
    "begin_capture",
    "capture_is_piecewise",
    "finish_capture",
    "full_graphs",
    "parse_enabled_hooks",
    "step_is_gated",
]
