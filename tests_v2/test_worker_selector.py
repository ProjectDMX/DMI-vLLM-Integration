"""Isolation checks for the dynamic DMI worker entry point."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from dmi_vllm_integration.adapter import DMXGPUWorker as DMXV1GPUWorker
from dmi_vllm_integration.worker import DMXGPUWorker


def _config(*, use_v2: bool) -> SimpleNamespace:
    return SimpleNamespace(use_v2_model_runner=use_v2)


def test_v1_selection_does_not_import_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in tuple(sys.modules):
        if module_name.startswith("dmi_vllm_integration.v2"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    worker = DMXGPUWorker.__new__(
        DMXGPUWorker,
        vllm_config=_config(use_v2=False),
    )

    assert isinstance(worker, DMXGPUWorker)
    assert isinstance(worker, DMXV1GPUWorker)
    assert not any(
        module_name.startswith("dmi_vllm_integration.v2")
        for module_name in sys.modules
    )


def test_v1_selection_preserves_public_worker_subclasses() -> None:
    class DerivedV1Worker(DMXGPUWorker):
        marker = object()

    worker = DerivedV1Worker.__new__(
        DerivedV1Worker,
        vllm_config=_config(use_v2=False),
    )

    assert isinstance(worker, DerivedV1Worker)
    assert worker.marker is DerivedV1Worker.marker


def test_v2_selection_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    from dmi_vllm_integration.v2.worker import DMXV2GPUWorker

    sentinel = object()
    monkeypatch.setattr(
        DMXV2GPUWorker,
        "__new__",
        staticmethod(lambda cls, *args, **kwargs: sentinel),
    )

    assert (
        DMXGPUWorker.__new__(
            DMXGPUWorker,
            vllm_config=_config(use_v2=True),
        )
        is sentinel
    )


def test_v2_selection_rejects_dynamic_entry_subclasses() -> None:
    class DerivedWorker(DMXGPUWorker):
        pass

    with pytest.raises(RuntimeError, match="subclass DMXV2GPUWorker directly"):
        DerivedWorker.__new__(
            DerivedWorker,
            vllm_config=_config(use_v2=True),
        )
