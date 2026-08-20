"""Opt-in vLLM endpoint for authoritative online DMI finalization."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.datastructures import State

from .compat import require_compatible_runtime

STOP_MONITORING_PATH = "/v1/dmi/stop_monitoring"
_STATE_KEY = "dmi_stop_monitoring"


class _StopMonitoringController:
    def __init__(self, engine_client: Any | None) -> None:
        self._engine_client = engine_client
        self._lock = asyncio.Lock()
        self._stopped = False

    async def stop(self, timeout: float) -> None:
        if self._engine_client is None:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="DMI finalization requires a vLLM engine client.",
            )

        async with self._lock:
            if self._stopped:
                return

            # Keep the scheduler paused after finalization. New work must not
            # execute after workers close their terminal DMI engines.
            await self._engine_client.pause_generation(
                mode="wait",
                clear_cache=False,
            )
            await self._engine_client.collective_rpc(
                method="stop_monitoring",
                timeout=timeout,
            )
            self._stopped = True


class DMIStopMonitoringEndpointPlugin:
    """Add a terminal, explicitly allowlisted online-finalization endpoint."""

    name = "dmi_stop_monitoring"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        async def stop_monitoring(
            raw_request: Request,
            timeout: Annotated[float, Query(gt=0)] = 30.0,
        ) -> dict[str, str]:
            controller = getattr(raw_request.app.state, _STATE_KEY, None)
            if controller is None:
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    detail="DMI finalization endpoint is not initialized.",
                )
            try:
                await controller.stop(timeout)
            except HTTPException:
                raise
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=HTTPStatus.GATEWAY_TIMEOUT,
                    detail="Timed out while stopping DMI monitoring.",
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="Failed to stop DMI monitoring.",
                ) from exc
            return {"status": "stopped"}

        app.add_api_route(
            STOP_MONITORING_PATH,
            stop_monitoring,
            methods=["POST"],
            name="dmi_stop_monitoring",
        )

    async def init_state(
        self,
        engine_client: Any | None,
        state: State,
        args: Any,
    ) -> None:
        del args
        require_compatible_runtime()
        setattr(state, _STATE_KEY, _StopMonitoringController(engine_client))


__all__ = ["DMIStopMonitoringEndpointPlugin", "STOP_MONITORING_PATH"]
