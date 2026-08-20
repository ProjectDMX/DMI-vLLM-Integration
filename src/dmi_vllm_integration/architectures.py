"""Supported model-architecture boundary for the vLLM integration."""

from __future__ import annotations

from typing import Any


LLAMA_COMPAT_ARCHITECTURES = frozenset(
    {
        "AquilaModel",
        "AquilaForCausalLM",
        "CwmForCausalLM",
        "InternLMForCausalLM",
        "InternLM3ForCausalLM",
        "IQuestCoderForCausalLM",
        "LlamaForCausalLM",
        "LLaMAForCausalLM",
        "XverseForCausalLM",
    }
)


ARCHITECTURE_REMAP: dict[str, str] = {
    "ApertusForCausalLM": "DMIApertusForCausalLM",
    "Ernie4_5ForCausalLM": "DMIErnie4_5ForCausalLM",
    "FalconH1ForCausalLM": "DMIFalconH1ForCausalLM",
    "Gemma3ForCausalLM": "DMIGemma3ForCausalLM",
    "GlmMoeDsaForCausalLM": "DMIGlmMoeDsaForCausalLM",
    "GPT2LMHeadModel": "DMIGPT2LMHeadModel",
    "GraniteForCausalLM": "DMIGraniteForCausalLM",
    "JambaForCausalLM": "DMIJambaForCausalLM",
    "Lfm2ForCausalLM": "DMILfm2ForCausalLM",
    "MiniCPMForCausalLM": "DMIMiniCPMForCausalLM",
    "MistralForCausalLM": "DMIMistralForCausalLM",
    "Olmo3ForCausalLM": "DMIOlmo3ForCausalLM",
    "Phi3ForCausalLM": "DMIPhi3ForCausalLM",
    "Qwen2ForCausalLM": "DMIQwen2ForCausalLM",
    "Qwen2MoeForCausalLM": "DMIQwen2MoeForCausalLM",
    "Qwen3ForCausalLM": "DMIQwen3ForCausalLM",
    **{
        architecture: "DMILlamaForCausalLM"
        for architecture in LLAMA_COMPAT_ARCHITECTURES
    },
}
SUPPORTED_CONFIG_ARCHITECTURES = frozenset(
    (*ARCHITECTURE_REMAP, *ARCHITECTURE_REMAP.values())
)


def require_supported_architecture(model_config: Any) -> tuple[str, ...]:
    """Validate and return the declared Hugging Face architectures."""

    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or not architectures
        or any(not isinstance(arch, str) for arch in architectures)
    ):
        raise RuntimeError(
            "DMI vLLM requires the model config to declare a supported "
            "architecture"
        )
    if not any(
        arch in SUPPORTED_CONFIG_ARCHITECTURES for arch in architectures
    ):
        supported = ", ".join(sorted(ARCHITECTURE_REMAP))
        configured = ", ".join(architectures)
        raise RuntimeError(
            "DMI vLLM does not support the configured model "
            f"architecture(s): {configured}. Supported architectures: "
            f"{supported}"
        )
    return tuple(architectures)


__all__ = [
    "LLAMA_COMPAT_ARCHITECTURES",
    "ARCHITECTURE_REMAP",
    "SUPPORTED_CONFIG_ARCHITECTURES",
    "require_supported_architecture",
]
