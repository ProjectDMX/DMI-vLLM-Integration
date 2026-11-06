"""In-process sinks, deferred request finalize and the worker RPCs.

The worker is assembled without vLLM (``DMXGPUWorker.__new__``) around a
fake ring engine whose delivery barrier outcome each test chooses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmi_vllm_integration import sinks
from dmi_vllm_integration.adapter import DMXGPUWorker, normalize_vllm_request_id
from dmi_vllm_integration.capture_selection import CaptureSelection


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def _clean_factory():
    sinks.clear_sink_factory()
    yield
    sinks.clear_sink_factory()


def test_sink_factory_registry():
    assert sinks.make_registered_sink() is None
    with pytest.raises(TypeError):
        sinks.register_sink_factory(object())
    sink = object()
    sinks.register_sink_factory(lambda: sink)
    assert sinks.make_registered_sink() is sink
    sinks.register_sink_factory(lambda: None)
    with pytest.raises(RuntimeError, match="returned None"):
        sinks.make_registered_sink()
    sinks.clear_sink_factory()
    assert sinks.registered_sink_factory() is None


def test_ring_consumer_prefers_a_native_queue():
    queue = object()
    assert sinks.ring_consumer_for(SimpleNamespace(ring_sink=queue)) is queue
    plain = SimpleNamespace(ring_sink=None)
    assert sinks.ring_consumer_for(plain) is plain
    bare = object()
    assert sinks.ring_consumer_for(bare) is bare


def test_adapter_module_reexports_the_registry():
    from dmi_vllm_integration import adapter
    assert adapter.register_sink_factory is sinks.register_sink_factory
    assert adapter.clear_sink_factory is sinks.clear_sink_factory


class Engine:
    """Ring engine double: a scripted delivery barrier and failure counters."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.submit_exceptions = 0
        self.sink_failed = False
        self.last_sink_error = ""
        self.barriers = []

    def drain_to_sink_and_wait(self, timeout_ms):
        self.barriers.append(timeout_ms)
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if outcome == "raise":
            self.submit_exceptions += 1
            return True
        return outcome

    def get_stats(self):
        return SimpleNamespace()

    def payload_cap(self):
        return 1000

    def available_capacity(self):
        return 400

    def staging_cap(self):
        return 2000


class Sink:
    def __init__(self, fail=()):
        self.finalized = []
        self.fail = set(fail)

    def finalize_request(self, rid, valid=True):
        if rid in self.fail:
            self.fail.discard(rid)
            raise RuntimeError("busy")
        self.finalized.append((rid, valid))

    def drop_stats(self):
        return {"dropped": 0}


def _worker(engine, sink=None, *, steps=3):
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.adaptor = SimpleNamespace(ring_engine=engine, selection=CaptureSelection(),
                                     _step_request_identities={})
    worker._dmx_sink = sink if sink is not None else Sink()
    worker._dmx_auto_finalize = True
    worker._dmx_pending_finalize = {}
    worker._dmx_live_requests = {}
    worker._dmx_finalize_max_steps = steps
    worker._dmx_finalize_drain_timeout_ms = 20
    worker._dmx_finalize_sweep_timeout_ms = 50
    worker._dmx_finalize_timeouts = 0
    worker._dmx_finalize_waits = []
    worker._dmx_stopped = False
    worker._dmx_host_engine = None
    return worker


def _finished(*ids):
    return SimpleNamespace(finished_req_ids=set(ids))


def test_drain_to_sink_statuses():
    engine = Engine([True, False, "raise"])
    worker = _worker(engine)
    assert worker.dmx_drain_to_sink(5) == 0
    assert worker.dmx_drain_to_sink(5) == 1
    assert worker.dmx_drain_to_sink(5) == 2          # a new consumer exception
    engine.sink_failed = True
    assert worker.dmx_drain_to_sink(5) == 2          # latched failure
    worker.adaptor = None
    assert worker.dmx_drain_to_sink(5) == -1


def test_finished_requests_finalize_after_the_delivery_barrier():
    engine = Engine([True])
    worker = _worker(engine)
    worker._dmx_finalize_finished(_finished("r1", "r2"))
    assert sorted(worker._dmx_sink.finalized) == [("r1", True), ("r2", True)]
    assert worker._dmx_pending_finalize == {}
    assert engine.barriers == [20]
    worker._dmx_finalize_finished(_finished())         # nothing pending: no barrier
    assert engine.barriers == [20]


def test_barrier_timeouts_retry_then_finalize_invalid():
    engine = Engine([False, False, False])
    worker = _worker(engine, steps=2)
    worker._dmx_finalize_finished(_finished("r1"))
    assert worker._dmx_sink.finalized == [] and worker._dmx_pending_finalize == {"r1": 1}
    worker._dmx_finalize_finished(_finished())
    assert worker._dmx_sink.finalized == [("r1", False)]
    assert worker._dmx_finalize_timeouts == 2 and len(worker._dmx_finalize_waits) == 2


def test_a_consumer_failure_finalizes_invalid_at_once():
    worker = _worker(Engine(["raise"]))
    worker._dmx_finalize_finished(_finished("r1"))
    assert worker._dmx_sink.finalized == [("r1", False)]


def test_masked_requests_finalize_invalid_and_rejected_finalizes_retry():
    sink = Sink(fail={"r2"})
    worker = _worker(Engine([True, True]), sink)
    worker.adaptor.selection._tainted.add("r1")
    worker._dmx_finalize_finished(_finished("r1", "r2"))
    assert sink.finalized == [("r1", False)]
    assert not worker.adaptor.selection.is_tainted("r1")
    assert worker._dmx_pending_finalize == {"r2": 3}  # kept for the retry
    worker._dmx_finalize_finished(_finished())
    assert sink.finalized[-1] == ("r2", True)


def test_a_notice_for_a_reused_id_does_not_finalize_the_new_request(monkeypatch):
    from dmi_vllm_integration import adapter
    monkeypatch.setattr(adapter, "normalize_vllm_request_id",
                        lambda rid: rid.rsplit("-", 1)[0])
    worker = _worker(Engine())
    worker.adaptor._step_request_identities = {"job": "job-1234abcd"}
    worker._dmx_finalize_finished(_finished("job-9999ffff"))   # earlier use of "job"
    assert worker._dmx_sink.finalized == []
    worker._dmx_finalize_finished(_finished("job-1234abcd"))
    assert worker._dmx_sink.finalized == [("job", True)]


def test_stop_delivers_sweeps_and_is_retryable_on_timeout():
    engine = Engine([False])
    worker = _worker(engine)
    worker._dmx_pending_finalize = {"r1": 2}
    closed = []
    worker.adaptor.close = lambda: closed.append(True)
    worker.adaptor.capture_dispatch_state = lambda: {}
    with pytest.raises(TimeoutError, match="tail data"):
        worker.stop_monitoring()
    assert worker._dmx_stopped is False and closed == []
    worker.stop_monitoring()                           # barrier succeeds now
    assert worker._dmx_sink.finalized == [("r1", True)]
    assert closed == [True] and worker.adaptor is None
    assert worker.dmx_sink_stats()["drops"] == {"dropped": 0}
    worker.stop_monitoring()                           # reentrant


def test_rpcs_delegate_to_the_capture_selection():
    worker = _worker(Engine())
    state = worker.dmx_set_request_mask(["a-0123abcd"])
    assert state["allowlist"] == [normalize_vllm_request_id("a-0123abcd")]
    worker.adaptor.capture_dispatch_state = lambda: {"prepared_steps": 3}
    assert worker.dmx_request_mask_state()["capture_dispatch"] == {"prepared_steps": 3}
    with pytest.raises(RuntimeError, match="before graph capture"):
        worker.dmx_set_hook_request_selection({})
    with pytest.raises(RuntimeError, match="not installed"):
        worker.dmx_inline_register([])
    assert worker.dmx_ring_stats() == {"payload_cap_bytes": 1000, "available_bytes": 400,
                                       "occupancy_bytes": 600, "staging_cap_bytes": 2000}
    stats = worker.dmx_sink_stats()
    assert stats["submit_exceptions"] == 0 and stats["sink_failed"] is False
    worker.adaptor = None
    with pytest.raises(RuntimeError, match="not initialized"):
        worker.dmx_set_request_mask(None)


def test_explicit_completion_and_the_stop_sweep_of_unreported_requests():
    worker = _worker(Engine([True, True]))
    worker.adaptor._step_request_identities = {"r1": "r1", "r2": "r2"}
    worker._dmx_finalize_finished(_finished())          # observed, not finished
    reply = worker.dmx_finalize_requests(["r1"])
    assert reply == {"status": 0, "pending": []}
    assert worker._dmx_sink.finalized == [("r1", True)]
    worker.adaptor.close = lambda: None
    worker.stop_monitoring()
    # r2 never reported finished: its completion is unknown.
    assert worker._dmx_sink.finalized == [("r1", True), ("r2", False)]
    worker._dmx_auto_finalize = False
    with pytest.raises(RuntimeError, match="does not finalize"):
        worker.dmx_finalize_requests(["x"])


def test_the_v2_entry_point_refuses_v1_only_features():
    from dmi_vllm_integration.worker import _refuse_v1_only_features

    def config(**settings):
        return SimpleNamespace(additional_config=settings)

    _refuse_v1_only_features(config())
    _refuse_v1_only_features(config(dmx_resid_pre_read="add", dmx_node_toggle=False,
                                    dmx_monitor_capture_mode="dynamic"))
    with pytest.raises(RuntimeError, match="dmx_node_toggle.*V1 model runner"):
        _refuse_v1_only_features(config(dmx_node_toggle=True))
    sinks.register_sink_factory(object)
    with pytest.raises(RuntimeError, match="register_sink_factory"):
        _refuse_v1_only_features(config())
