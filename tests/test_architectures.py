"""Portable CPU tests for the supported model-architecture boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmi_vllm_integration.architectures import (
    ARCHITECTURE_REMAP,
    require_supported_architecture,
)
from dmi_vllm_integration.plugin import MODEL_REGISTRATIONS


def _model_config(architectures):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=architectures)
    )


@pytest.mark.parametrize(
    "architecture",
    [*ARCHITECTURE_REMAP, *ARCHITECTURE_REMAP.values()],
)
def test_declared_supported_architecture_is_accepted(architecture: str) -> None:
    assert require_supported_architecture(
        _model_config([architecture])
    ) == (architecture,)


def test_supported_aliases_are_exactly_the_plugin_registrations() -> None:
    assert set(ARCHITECTURE_REMAP.values()) == set(MODEL_REGISTRATIONS)


@pytest.mark.parametrize(
    "architectures",
    [None, [], "LlamaForCausalLM", [None], ["MambaForCausalLM"]],
)
def test_missing_malformed_or_unsupported_architecture_is_rejected(
    architectures,
) -> None:
    with pytest.raises(RuntimeError, match="(?i)supported.*architecture"):
        require_supported_architecture(_model_config(architectures))


def test_a_supported_fallback_in_the_declared_list_is_accepted() -> None:
    assert require_supported_architecture(
        _model_config(["UnknownArchitecture", "Qwen3ForCausalLM"])
    ) == ("UnknownArchitecture", "Qwen3ForCausalLM")
