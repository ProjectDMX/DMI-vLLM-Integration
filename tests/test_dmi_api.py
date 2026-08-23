from __future__ import annotations

from types import ModuleType

import pytest

from dmi_vllm_integration import dmi_api


def test_selected_dmi_api_exports_version_one() -> None:
    assert dmi_api.DMI_INTEGRATION_API_VERSION == 1


def test_dmi_api_prefers_canonical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = ModuleType(dmi_api.CANONICAL_DMI_API_MODULE)
    calls: list[str] = []

    def import_module(name: str) -> ModuleType:
        calls.append(name)
        return canonical

    monkeypatch.setattr(dmi_api, "import_module", import_module)

    assert dmi_api._import_dmi_api_v1() is canonical
    assert calls == [dmi_api.CANONICAL_DMI_API_MODULE]


def test_dmi_api_falls_back_when_dmi_package_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType(dmi_api.LEGACY_DMI_API_MODULE)
    calls: list[str] = []

    def import_module(name: str) -> ModuleType:
        calls.append(name)
        if name == dmi_api.CANONICAL_DMI_API_MODULE:
            raise ModuleNotFoundError("No module named 'dmi'", name="dmi")
        return legacy

    monkeypatch.setattr(dmi_api, "import_module", import_module)

    assert dmi_api._import_dmi_api_v1() is legacy
    assert calls == [
        dmi_api.CANONICAL_DMI_API_MODULE,
        dmi_api.LEGACY_DMI_API_MODULE,
    ]


def test_dmi_api_does_not_hide_canonical_internal_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def import_module(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(
            "No module named 'dmi_internal_dependency'",
            name="dmi_internal_dependency",
        )

    monkeypatch.setattr(dmi_api, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        dmi_api._import_dmi_api_v1()

    assert exc_info.value.name == "dmi_internal_dependency"
    assert calls == [dmi_api.CANONICAL_DMI_API_MODULE]


@pytest.mark.parametrize("missing_module", ["dmi.api", "dmi.api.v1"])
def test_dmi_api_does_not_fallback_for_partial_canonical_layout(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
) -> None:
    calls: list[str] = []

    def import_module(name: str) -> ModuleType:
        calls.append(name)
        raise ModuleNotFoundError(
            f"No module named '{missing_module}'",
            name=missing_module,
        )

    monkeypatch.setattr(dmi_api, "import_module", import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        dmi_api._import_dmi_api_v1()

    assert exc_info.value.name == missing_module
    assert calls == [dmi_api.CANONICAL_DMI_API_MODULE]
