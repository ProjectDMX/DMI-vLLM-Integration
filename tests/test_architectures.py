"""Portable CPU tests for the supported model-architecture boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmi_vllm_integration.architectures import (
    ARCHITECTURE_REMAP,
    require_supported_architecture,
)
from dmi_vllm_integration.plugin import MODEL_REGISTRATIONS


OBSOLETE_VLLM_ALIASES = {
    "AquilaModel",
    "AquilaForCausalLM",
    "InternLMForCausalLM",
    "XverseForCausalLM",
}


def _model_config(architectures, *, resolved=None):
    if resolved is None and isinstance(architectures, list) and architectures:
        resolved = architectures[0]
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=architectures),
        architecture=resolved,
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


@pytest.mark.parametrize("architecture", sorted(OBSOLETE_VLLM_ALIASES))
def test_alias_removed_from_vllm_027_is_rejected(architecture: str) -> None:
    assert architecture not in ARCHITECTURE_REMAP
    with pytest.raises(RuntimeError, match="resolved by vLLM"):
        require_supported_architecture(_model_config([architecture]))


@pytest.mark.parametrize(
    "architectures",
    [None, [], "LlamaForCausalLM", [None], ["MambaForCausalLM"]],
)
def test_missing_malformed_or_unsupported_architecture_is_rejected(
    architectures,
) -> None:
    with pytest.raises(RuntimeError, match="(?i)supported.*architecture"):
        require_supported_architecture(_model_config(architectures))


def test_vllm_resolved_architecture_wins_over_a_supported_later_entry() -> None:
    with pytest.raises(RuntimeError, match="resolved by vLLM"):
        require_supported_architecture(
            _model_config(
                ["UpstreamResolvableArchitecture", "Qwen3ForCausalLM"],
                resolved="UpstreamResolvableArchitecture",
            )
        )


def test_supported_resolved_fallback_is_accepted() -> None:
    assert require_supported_architecture(
        _model_config(
            ["UnknownArchitecture", "Qwen3ForCausalLM"],
            resolved="Qwen3ForCausalLM",
        )
    ) == ("Qwen3ForCausalLM",)
