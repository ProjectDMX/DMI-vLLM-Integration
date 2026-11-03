"""In-process consumers for the monitored vLLM worker.

vLLM constructs ``DMXGPUWorker`` itself and passes it only serializable
configuration, so a live Python consumer cannot travel in
``additional_config``.  An application instead registers a factory here
before it constructs the engine; ``DMXGPUWorker.init_device`` calls it once
and wires the returned object into the ring.  The registry is process-local:
the worker must run in the registering process
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``), or the application must register
from inside its own worker subclass.

A registered sink is any object.  The worker recognizes these optional
attributes:

``ring_sink``
    The native ring consumer (``dmi.transport.native.SliceQueue`` or
    ``BatchQueue``) the drain thread pushes to without the GIL.  Absent or
    ``None``: the sink object itself is the consumer and is called per
    request slice (``(model_id, shard_rank, req_id, act_name, layer_no,
    start_token, end_token, tensor)``).
``dmx_batch_submit``
    True selects the whole-hook batch handoff (requires a ``BatchQueue``).
``finalize_request(req_id, valid=True)``
    Called once per finished request after its last captured rows reached
    the consumer (``dmx_auto_finalize``).
``observe_request_context(prompt_lengths)``
    Called each step with ``{request_id: prompt_length}`` for the step's
    requests, before capture is planned.
``capture_planner``
    An object that plans per-request capture each step (see
    ``capture_selection.CapturePlanner``), or ``None``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


_SINK_FACTORY: Optional[Callable[[], Any]] = None


def register_sink_factory(factory: Callable[[], Any]) -> None:
    """Install the in-process consumer factory for the next worker.

    The factory is called once per worker, from ``init_device``.  The caller
    owns the returned object's lifetime: the worker never starts or stops it.
    """
    if not callable(factory):
        raise TypeError("sink factory must be callable")
    global _SINK_FACTORY
    _SINK_FACTORY = factory


def clear_sink_factory() -> None:
    """Remove the registered factory (workers already built keep theirs)."""
    global _SINK_FACTORY
    _SINK_FACTORY = None


def registered_sink_factory() -> Optional[Callable[[], Any]]:
    return _SINK_FACTORY


def make_registered_sink() -> Any:
    """Call the registered factory; ``None`` when none is registered."""
    factory = _SINK_FACTORY
    if factory is None:
        return None
    sink = factory()
    if sink is None:
        raise RuntimeError("the registered DMI sink factory returned None")
    return sink


def ring_consumer_for(sink: Any) -> Any:
    """The object the ring delivers to: ``sink.ring_sink`` when set, else
    the sink itself.  A sink may carry ``ring_sink = None`` (no native
    queue); that falls back to the sink rather than running sink-less."""
    consumer = getattr(sink, "ring_sink", None)
    return sink if consumer is None else consumer


__all__ = [
    "clear_sink_factory",
    "make_registered_sink",
    "register_sink_factory",
    "registered_sink_factory",
    "ring_consumer_for",
]
