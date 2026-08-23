"""Pre-device checks for concrete DMI model implementation boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .architectures import ARCHITECTURE_REMAP


@dataclass(frozen=True)
class _ValidationSpec:
    """Lazy call specification for one DMI-specific model validator."""

    module: str
    validator: str
    positional_args: tuple[str, ...] = ("hf_config",)
    keyword_args: tuple[str, ...] = ()


# Only checks that protect a concrete DMI implementation invariant belong
# here. Generic model/config validity and release-cell policy remain upstream
# or in release documentation.
_UPSTREAM_VALIDATION_SPECS = {
    "MiniCPMForCausalLM": _ValidationSpec(
        "dmi_vllm_integration.models.minicpm",
        "_require_supported_minicpm_config",
    ),
    "MistralForCausalLM": _ValidationSpec(
        "dmi_vllm_integration.models.mistral",
        "_reject_unsupported_mistral_branches",
        positional_args=("vllm_config",),
    ),
    "Qwen3MoeForCausalLM": _ValidationSpec(
        "dmi_vllm_integration.models.qwen3_moe",
        "_require_supported_qwen3_moe_config",
    ),
    "Qwen3_5ForConditionalGeneration": _ValidationSpec(
        "dmi_vllm_integration.models.qwen3_5",
        "_require_supported_qwen36_config",
    ),
    "GlmMoeDsaForCausalLM": _ValidationSpec(
        "dmi_vllm_integration.models.glm_moe_dsa",
        "_require_supported_glm52_config",
        keyword_args=("dtype", "use_mla"),
    ),
    "Gemma4ForConditionalGeneration": _ValidationSpec(
        "dmi_vllm_integration.models.gemma4",
        "_require_supported_gemma4_e2b_config",
        positional_args=("hf_config", "parallel_config"),
        keyword_args=("kv_sharing_fast_prefill",),
    ),
    "Llama4ForConditionalGeneration": _ValidationSpec(
        "dmi_vllm_integration.models.llama4",
        "_require_supported_llama4_scout_config",
    ),
}


_VALIDATION_SPECS = dict(_UPSTREAM_VALIDATION_SPECS)
for _upstream, _spec in _UPSTREAM_VALIDATION_SPECS.items():
    _VALIDATION_SPECS[ARCHITECTURE_REMAP[_upstream]] = _spec


def _runtime_value(vllm_config: Any, name: str) -> Any:
    model_config = vllm_config.model_config
    if name == "vllm_config":
        return vllm_config
    if name == "hf_config":
        return model_config.hf_config
    if name == "parallel_config":
        return vllm_config.parallel_config
    if name == "dtype":
        return model_config.dtype
    if name == "use_mla":
        return model_config.use_mla
    if name == "kv_sharing_fast_prefill":
        return vllm_config.cache_config.kv_sharing_fast_prefill
    raise AssertionError(  # pragma: no cover - guarded by the static table
        f"unknown model-validation argument: {name}"
    )


def validate_model_specific_config(
    vllm_config: Any,
    architectures: Iterable[str],
) -> None:
    """Fail before CUDA only for a selected DMI implementation boundary.

    Models without a DMI-specific boundary are intentionally absent. Their
    ordinary configuration validity remains owned by official vLLM.
    """

    selected = next(
        (
            _VALIDATION_SPECS[architecture]
            for architecture in architectures
            if architecture in _VALIDATION_SPECS
        ),
        None,
    )
    if selected is None:
        return

    validator = getattr(import_module(selected.module), selected.validator)
    validator(
        *(_runtime_value(vllm_config, name) for name in selected.positional_args),
        **{name: _runtime_value(vllm_config, name) for name in selected.keyword_args},
    )


__all__ = ["validate_model_specific_config"]
