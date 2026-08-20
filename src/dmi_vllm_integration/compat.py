"""Fail-fast runtime compatibility checks for the integration package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

INTEGRATION_DISTRIBUTION = "DMI-vLLM-Integration"
VLLM_DISTRIBUTION = "vllm"
REQUIRED_DMI_API_VERSION = 1


class CompatibilityError(RuntimeError):
    """The installed DMI, vLLM, and integration versions cannot work together."""


@dataclass(frozen=True)
class RuntimeCompatibility:
    """Validated versions for the current integration process."""

    integration_version: str
    expected_vllm_version: str
    vllm_version: str
    dmi_api_version: int


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise CompatibilityError(
            f"Required distribution {distribution!r} is not installed."
        ) from exc


def _parse_version(raw_version: str, distribution: str) -> Version:
    try:
        return Version(raw_version)
    except InvalidVersion as exc:
        raise CompatibilityError(
            f"{distribution} reports invalid version {raw_version!r}."
        ) from exc


def _dmi_api_version() -> int:
    try:
        api = import_module("monitoring.integration_api.v1")
    except (ImportError, OSError) as exc:
        raise CompatibilityError(
            "DMI integration API v1 is unavailable; install DMI>=1.1.0,<2.0."
        ) from exc

    api_version = getattr(api, "DMI_INTEGRATION_API_VERSION", None)
    if not isinstance(api_version, int):
        raise CompatibilityError(
            "monitoring.integration_api.v1 does not expose an integer "
            "DMI_INTEGRATION_API_VERSION."
        )
    return api_version


def check_runtime_compatibility() -> RuntimeCompatibility:
    """Validate the official vLLM release and DMI integration API version.

    Integration post-releases continue to target the vLLM release encoded by
    their base version. Local vLLM build metadata is accepted, but prerelease,
    development, and post-release versions are not treated as that release.
    """

    integration_raw = _distribution_version(INTEGRATION_DISTRIBUTION)
    integration_version = _parse_version(integration_raw, INTEGRATION_DISTRIBUTION)
    expected_vllm = integration_version.base_version

    vllm_raw = _distribution_version(VLLM_DISTRIBUTION)
    vllm_version = _parse_version(vllm_raw, VLLM_DISTRIBUTION)
    if vllm_version.public != expected_vllm:
        raise CompatibilityError(
            f"{INTEGRATION_DISTRIBUTION} {integration_raw} requires "
            f"vLLM {expected_vllm}; found {vllm_raw}."
        )

    dmi_api_version = _dmi_api_version()
    if dmi_api_version != REQUIRED_DMI_API_VERSION:
        raise CompatibilityError(
            f"{INTEGRATION_DISTRIBUTION} {integration_raw} requires DMI "
            f"integration API v{REQUIRED_DMI_API_VERSION}; found "
            f"v{dmi_api_version}."
        )

    return RuntimeCompatibility(
        integration_version=integration_raw,
        expected_vllm_version=expected_vllm,
        vllm_version=vllm_raw,
        dmi_api_version=dmi_api_version,
    )


def require_compatible_runtime() -> RuntimeCompatibility:
    """Require a supported runtime before vLLM initializes GPU execution."""

    return check_runtime_compatibility()


__all__ = [
    "CompatibilityError",
    "RuntimeCompatibility",
    "check_runtime_compatibility",
    "require_compatible_runtime",
]
