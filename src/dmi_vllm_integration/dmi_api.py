"""DMI integration API v1 import compatibility.

DMI 1.2 moved the unchanged API-v1 facade from
``monitoring.integration_api.v1`` to ``dmi.api.v1``. Prefer the canonical
location while retaining compatibility with DMI 1.1. An import failure raised
from inside the canonical facade is never hidden by the legacy fallback.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


CANONICAL_DMI_API_MODULE = "dmi.api.v1"
LEGACY_DMI_API_MODULE = "monitoring.integration_api.v1"


def _import_dmi_api_v1() -> ModuleType:
    try:
        return import_module(CANONICAL_DMI_API_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != "dmi":
            raise
    return import_module(LEGACY_DMI_API_MODULE)


_API = _import_dmi_api_v1()
__all__ = list(_API.__all__)


def __getattr__(name: str) -> Any:
    return getattr(_API, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_API)))
