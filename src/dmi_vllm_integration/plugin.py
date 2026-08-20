"""vLLM general plugin for DMI-monitored model architectures."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_REGISTRATIONS: Mapping[str, str] = {
    "DMIApertusForCausalLM": (
        "dmi_vllm_integration.models.apertus:ApertusPForCausalLM"
    ),
    "DMIErnie4_5ForCausalLM": (
        "dmi_vllm_integration.models.ernie45:Ernie4_5PForCausalLM"
    ),
    "DMIFalconH1ForCausalLM": (
        "dmi_vllm_integration.models.falcon_h1:FalconH1PForCausalLM"
    ),
    "DMIGemma3ForCausalLM": (
        "dmi_vllm_integration.models.gemma3:Gemma3PForCausalLM"
    ),
    "DMIGlmMoeDsaForCausalLM": (
        "dmi_vllm_integration.models.glm_moe_dsa:GlmMoeDsaPForCausalLM"
    ),
    "DMIGPT2LMHeadModel": (
        "dmi_vllm_integration.models.gpt2:GPT2PLMHeadModel"
    ),
    "DMIGraniteForCausalLM": (
        "dmi_vllm_integration.models.granite:GranitePForCausalLM"
    ),
    "DMIJambaForCausalLM": (
        "dmi_vllm_integration.models.jamba:JambaPForCausalLM"
    ),
    "DMILfm2ForCausalLM": (
        "dmi_vllm_integration.models.lfm2:Lfm2PForCausalLM"
    ),
    "DMILlamaForCausalLM": (
        "dmi_vllm_integration.models.llama:LlamaPForCausalLM"
    ),
    "DMIMiniCPMForCausalLM": (
        "dmi_vllm_integration.models.minicpm:MiniCPMPForCausalLM"
    ),
    "DMIMistralForCausalLM": (
        "dmi_vllm_integration.models.mistral:MistralPForCausalLM"
    ),
    "DMIOlmo3ForCausalLM": (
        "dmi_vllm_integration.models.olmo3:Olmo3PForCausalLM"
    ),
    "DMIPhi3ForCausalLM": (
        "dmi_vllm_integration.models.phi3:Phi3PForCausalLM"
    ),
    "DMIQwen2ForCausalLM": (
        "dmi_vllm_integration.models.qwen2:Qwen2PForCausalLM"
    ),
    "DMIQwen2MoeForCausalLM": (
        "dmi_vllm_integration.models.qwen2_moe:Qwen2MoePForCausalLM"
    ),
    "DMIQwen3ForCausalLM": (
        "dmi_vllm_integration.models.qwen3:Qwen3PForCausalLM"
    ),
}


def register() -> None:
    """Validate the runtime, then assert DMI's package-owned aliases."""

    from dmi_vllm_integration.compat import require_compatible_runtime

    require_compatible_runtime()

    from vllm import ModelRegistry

    for architecture, target in MODEL_REGISTRATIONS.items():
        # These names are DMI-owned aliases. Re-registering the same lazy
        # targets is idempotent, and deterministically replaces a conflicting
        # third-party registration instead of silently remapping to it.
        ModelRegistry.register_model(architecture, target)


def register_models() -> None:
    """Compatibility spelling for direct callers; prefer :func:`register`."""

    register()
