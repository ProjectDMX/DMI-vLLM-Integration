from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest
from fastapi import FastAPI
from starlette.datastructures import State

from dmi_vllm_integration import compat, endpoint


def _versions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    integration: str = "0.27.1",
    vllm: str = "0.27.1",
    dmi_api: int = 1,
) -> None:
    versions = {
        compat.INTEGRATION_DISTRIBUTION: integration,
        compat.VLLM_DISTRIBUTION: vllm,
    }
    monkeypatch.setattr(compat, "_distribution_version", versions.__getitem__)
    monkeypatch.setattr(compat, "_dmi_api_version", lambda: dmi_api)


def test_runtime_compatibility_accepts_matching_base_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _versions(
        monkeypatch,
        integration="0.27.1.post2",
        vllm="0.27.1+cu130",
    )

    result = compat.check_runtime_compatibility()

    assert result.expected_vllm_version == "0.27.1"
    assert result.vllm_version == "0.27.1+cu130"
    assert result.dmi_api_version == 1


@pytest.mark.parametrize(
    ("integration_version", "vllm_version"),
    [
        ("0.27.1", "0.27.0"),
        ("0.27.1", "0.27.1rc1"),
        ("0.27.1.post1", "0.27.1.post1"),
    ],
)
def test_runtime_compatibility_rejects_other_vllm_releases(
    monkeypatch: pytest.MonkeyPatch,
    integration_version: str,
    vllm_version: str,
) -> None:
    _versions(
        monkeypatch,
        integration=integration_version,
        vllm=vllm_version,
    )

    with pytest.raises(compat.CompatibilityError, match="requires vLLM 0.27.1"):
        compat.check_runtime_compatibility()


def test_runtime_compatibility_requires_api_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _versions(monkeypatch, dmi_api=2)

    with pytest.raises(compat.CompatibilityError, match="requires DMI integration API v1"):
        compat.check_runtime_compatibility()


class _EngineClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def pause_generation(self, **kwargs) -> None:
        self.calls.append(("pause_generation", kwargs))

    async def collective_rpc(self, **kwargs) -> list[None]:
        self.calls.append(("collective_rpc", kwargs))
        return [None, None]


def test_endpoint_drains_then_stops_every_worker_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint, "require_compatible_runtime", lambda: None)
    plugin = endpoint.DMIStopMonitoringEndpointPlugin()
    app = FastAPI()
    plugin.attach_router(app)
    assert any(route.path == endpoint.STOP_MONITORING_PATH for route in app.routes)

    engine_client = _EngineClient()
    state = State()

    async def finalize_twice() -> None:
        await plugin.init_state(engine_client, state, Namespace())
        controller = getattr(state, "dmi_stop_monitoring")
        await controller.stop(12.5)
        await controller.stop(12.5)

    asyncio.run(finalize_twice())

    assert engine_client.calls == [
        (
            "pause_generation",
            {"mode": "wait", "clear_cache": False},
        ),
        (
            "collective_rpc",
            {"method": "stop_monitoring", "timeout": 12.5},
        ),
    ]
