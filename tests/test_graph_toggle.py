"""vLLM wiring of DMI's CUDA-graph node toggling (``graph_toggle``).

Graphs and the ring engine are fakes; the registry mechanics themselves are
DMI's (``dmi.node_toggle``) and tested there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

node_toggle = pytest.importorskip("dmi.node_toggle")

from vllm.compilation.cuda_graph import CUDAGraphWrapper  # noqa: E402
from vllm.config import CUDAGraphMode  # noqa: E402

from dmi_vllm_integration import graph_toggle  # noqa: E402
from dmi_vllm_integration.models.residual import (  # noqa: E402
    RESID_PRE_READ_MODES,
    resid_pre_from_norm,
)


class Graph:
    _dmi_keep_graph = True

    def __init__(self, handle):
        self.handle = handle

    def raw_cuda_graph(self):
        return self.handle

    def raw_cuda_graph_exec(self):
        return self.handle + 1


class Engine:
    def __init__(self):
        self.bound = []
        self.capture = None
        self.calls = []

    def clear_toggle_registry(self):
        self.calls.append("clear")

    def enable_device_toggle(self, on):
        self.calls.append(("device", on))

    def enable_toggle_capture(self, on):
        self.capture = on

    def toggle_capture_enabled(self):
        return bool(self.capture)

    def bind_graph_exec(self, graph, exec_):
        self.bound.append((graph, exec_))

    def bound_graph_count(self):
        return len(self.bound)

    def toggle_node_count(self):
        return 4 * len(self.bound)


class Transport:
    def __init__(self, engine):
        self._ring_engine = engine
        self.lazy = self.eager = None

    def set_active_hooks_lazy(self, enabled):
        self.lazy = enabled

    def set_active_hooks(self, enabled):
        self.eager = enabled


def _wrapper(mode, graphs, runnable=None):
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.__dict__.update(
        runtime_mode=mode, runnable=runnable,
        concrete_cudagraph_entries={i: SimpleNamespace(cudagraph=g)
                                    for i, g in enumerate(graphs)})
    return wrapper


@pytest.fixture(autouse=True)
def _guard_off():
    yield
    node_toggle.set_replay_guard(False)


def test_parse_and_step_classification():
    assert graph_toggle.parse_enabled_hooks("") is None
    assert graph_toggle.parse_enabled_hooks("none") == []
    assert graph_toggle.parse_enabled_hooks("0:3,17") == [(0, 3), (17, -1)]
    assert graph_toggle.step_is_gated(CUDAGraphMode.FULL) is True
    assert graph_toggle.step_is_gated(CUDAGraphMode.PIECEWISE) is False
    assert graph_toggle.step_is_gated(CUDAGraphMode.NONE) is False


def test_piecewise_detection_reads_the_forward_context(monkeypatch):
    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context",
                        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE))
    assert graph_toggle.capture_is_piecewise() is True
    monkeypatch.setattr(fc, "get_forward_context",
                        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL))
    assert graph_toggle.capture_is_piecewise() is False

    def missing():
        raise AssertionError("no forward context")
    monkeypatch.setattr(fc, "get_forward_context", missing)
    assert graph_toggle.capture_is_piecewise() is False


def test_only_full_graphs_are_found_through_nested_wrappers():
    full = [Graph(10), Graph(20)]
    piecewise = _wrapper(CUDAGraphMode.PIECEWISE, [Graph(30)])
    model = _wrapper(CUDAGraphMode.FULL, full, runnable=SimpleNamespace(runnable=piecewise))
    assert list(graph_toggle.full_graphs(model)) == full


def test_capture_window_binds_full_graphs_and_applies_the_enabled_set(monkeypatch):
    monkeypatch.setattr(node_toggle, "install_cudagraph_patch", lambda: None)
    engine = Engine()
    graph_toggle.begin_capture(engine, device_toggle=True)
    assert engine.capture is True and ("device", True) in engine.calls
    model = _wrapper(CUDAGraphMode.FULL, [Graph(10)])
    transport = Transport(engine)
    bound = graph_toggle.finish_capture(model, engine, transport, [(0, 1)], lazy=True)
    assert bound == 1 and engine.bound == [(10, 11)]
    assert engine.capture is False
    assert transport.lazy == [(0, 1)]
    assert node_toggle._REPLAY_GUARD is True


def test_an_enabled_set_without_a_full_graph_is_refused(monkeypatch):
    monkeypatch.setattr(node_toggle, "install_cudagraph_patch", lambda: None)
    engine = Engine()
    graph_toggle.begin_capture(engine)
    with pytest.raises(RuntimeError, match="FULL"):
        graph_toggle.finish_capture(_wrapper(CUDAGraphMode.PIECEWISE, [Graph(1)]),
                                    engine, Transport(engine), [(0, 1)], lazy=False)
    assert engine.capture is False                     # window closed regardless
    # No enabled set: serving continues with every hook on.
    graph_toggle.begin_capture(engine)
    assert graph_toggle.finish_capture(None, engine, Transport(engine), None,
                                       lazy=False) == 0


def test_abort_closes_the_window(monkeypatch):
    monkeypatch.setattr(node_toggle, "install_cudagraph_patch", lambda: None)
    engine = Engine()
    graph_toggle.begin_capture(engine)
    graph_toggle.abort_capture(engine)
    assert engine.capture is False and node_toggle._WINDOW.engine is None


def test_resid_pre_read_mode_outside_a_vllm_config_context():
    assert resid_pre_from_norm() is False


def test_resid_pre_read_mode():
    assert RESID_PRE_READ_MODES == ("add", "returned")
    assert resid_pre_from_norm(SimpleNamespace(additional_config={})) is False
    assert resid_pre_from_norm(SimpleNamespace(
        additional_config={"dmx_resid_pre_read": "returned"})) is True
    with pytest.raises(ValueError, match="dmx_resid_pre_read"):
        resid_pre_from_norm(SimpleNamespace(additional_config={"dmx_resid_pre_read": "late"}))


@pytest.mark.parametrize("module", ["llama", "qwen3"])
def test_returned_residual_is_observed_after_the_fused_norm(module):
    import ast
    import inspect
    import importlib

    source = inspect.getsource(importlib.import_module(
        f"dmi_vllm_integration.models.{module}"))
    forward = next(node for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.FunctionDef) and node.name == "forward"
                   and "hook_resid_pre" in ast.unparse(node))
    text = ast.unparse(forward)
    norm = text.index("self.input_layernorm(hidden_states, residual)")
    assert text.index("self.hook_resid_pre(hidden_states + residual)") < norm
    assert text.index("self.hook_resid_pre(residual)") > norm
    assert text.index("self.hook_resid_pre(residual)") < text.index("self.self_attn(")
