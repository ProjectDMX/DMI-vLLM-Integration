"""Contracts for Qwen2-MoE routing capture and validation oracles."""

from __future__ import annotations

import json
import shutil

import pytest
from transformers import Qwen2MoeConfig

from dmi_vllm_integration.adapter import _ARCH_REMAP
from dmi.api.v1 import (
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    compute_hook_shape,
    make_model_shape_from_hf_config,
)
from tests.oracles import register_oracle_models
from tests.oracles.enable_ref_hooks import enable_ref_hooks


pytestmark = pytest.mark.vllm


def _shape_config() -> Qwen2MoeConfig:
    return Qwen2MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=60,
        num_experts_per_tok=4,
        vocab_size=128,
    )


def test_routing_hook_shapes_follow_qwen2_moe_config() -> None:
    model_shape = make_model_shape_from_hf_config(_shape_config())
    assert model_shape is not None

    assert compute_hook_shape(
        HOOK_TYPE_ROUTER_LOGITS, model_shape, batch=0, q_len=17, kv_dim=17
    ) == [17, 60]
    assert compute_hook_shape(
        HOOK_TYPE_TOPK_IDS, model_shape, batch=0, q_len=17, kv_dim=17
    ) == [17, 4]
    assert compute_hook_shape(
        HOOK_TYPE_TOPK_WEIGHTS, model_shape, batch=0, q_len=17, kv_dim=17
    ) == [17, 4]


def test_adapter_remaps_qwen2_moe_to_unique_integration_alias() -> None:
    assert _ARCH_REMAP["Qwen2MoeForCausalLM"] == "DMIQwen2MoeForCausalLM"


def test_compare_and_ref_oracles_register_lazily() -> None:
    from vllm import ModelRegistry

    register_oracle_models()
    assert "DMIQwen2MoeCompareForCausalLM" in ModelRegistry.get_supported_archs()
    assert "DMIQwen2MoeRefForCausalLM" in ModelRegistry.get_supported_archs()


def test_qwen2_moe_ref_preset_adds_routing_hooks(tmp_path) -> None:
    source = tmp_path / "qwen2_moe_ref.py"
    shutil.copyfile(
        __file__.replace("test_moe_routing.py", "oracles/qwen2_moe_ref.py"),
        source,
    )
    output_dir = tmp_path / "out"
    config_path = tmp_path / "ref_config.json"

    enable_ref_hooks(
        model_file=str(source),
        hooks="vllm-full",
        max_len=128,
        output_dir=str(output_dir),
        config_out=str(config_path),
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert {"router_logits", "topk_ids", "topk_weights"} <= set(
        config["enabled_hooks"]
    )
    assert "pos_embed" not in config["enabled_hooks"]
    assert config["found_hooks"] == config["enabled_hooks"]
