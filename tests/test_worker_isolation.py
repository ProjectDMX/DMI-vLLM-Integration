"""Regression gates for the unchanged V1 worker implementation."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from dmi_vllm_integration.adapter import DMXGPUWorker as DMXV1GPUWorker
from dmi_vllm_integration.worker import DMXGPUWorker


def _v1_config() -> SimpleNamespace:
    return SimpleNamespace(use_v2_model_runner=False)


def test_v1_selection_does_not_import_v2() -> None:
    probe = """
import sys
from types import SimpleNamespace
import tests.conftest  # Install the opt-in hosted native contract stub.
from dmi_vllm_integration.adapter import DMXGPUWorker as V1
from dmi_vllm_integration.worker import DMXGPUWorker
worker = DMXGPUWorker.__new__(
    DMXGPUWorker,
    vllm_config=SimpleNamespace(use_v2_model_runner=False),
)
assert isinstance(worker, DMXGPUWorker)
assert isinstance(worker, V1)
assert not any(
    name.startswith("dmi_vllm_integration.v2") for name in sys.modules
)
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_v1_selection_preserves_public_worker_subclasses() -> None:
    class DerivedV1Worker(DMXGPUWorker):
        marker = object()

    worker = DerivedV1Worker.__new__(
        DerivedV1Worker,
        vllm_config=_v1_config(),
    )

    assert isinstance(worker, DerivedV1Worker)
    assert worker.marker is DerivedV1Worker.marker


def test_v1_direct_allocation_without_config_is_preserved() -> None:
    worker = DMXGPUWorker.__new__(DMXGPUWorker)

    assert isinstance(worker, DMXGPUWorker)
    assert isinstance(worker, DMXV1GPUWorker)
