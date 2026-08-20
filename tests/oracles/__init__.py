"""Lazy registration for the integration suite's Compare/Ref models.

These models are validation oracles, not production plugin registrations.
Runners opt into them explicitly before constructing a vLLM engine.
"""

from __future__ import annotations


_ORACLE_MODELS = {
    "DMIGPT2CompareLMHeadModel": "tests.oracles.gpt2_compare:GPT2CompareForCausalLM",
    "DMIGPT2RefLMHeadModel": "tests.oracles.gpt2_ref:GPT2RefLMHeadModel",
    "DMILlamaCompareForCausalLM": "tests.oracles.llama_compare:LlamaCompareForCausalLM",
    "DMILlamaRefForCausalLM": "tests.oracles.llama_ref:LlamaRefForCausalLM",
    "DMIQwen2MoeCompareForCausalLM": (
        "tests.oracles.qwen2_moe_compare:Qwen2MoeCompareForCausalLM"
    ),
    "DMIQwen2MoeRefForCausalLM": (
        "tests.oracles.qwen2_moe_ref:Qwen2MoeRefForCausalLM"
    ),
    "DMIQwen3CompareForCausalLM": (
        "tests.oracles.qwen3_compare:Qwen3CompareForCausalLM"
    ),
    "DMIQwen3RefForCausalLM": "tests.oracles.qwen3_ref:Qwen3RefForCausalLM",
}


def register_oracle_models() -> None:
    """Register validation-only architectures without importing model code."""

    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for architecture, model_path in _ORACLE_MODELS.items():
        if architecture not in supported:
            ModelRegistry.register_model(architecture, model_path)


__all__ = ["register_oracle_models"]
