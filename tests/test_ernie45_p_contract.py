"""CPU contracts for ERNIE 4.5 dense monitoring support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
from dmi_vllm_integration.dmi_api import HookPoint
from dmi_vllm_integration.dmi_api import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from vllm.model_executor.models.ernie45 import Ernie4_5ForCausalLM
import tests.oracles.llama_compare as llama_compare
from tests.oracles.ernie45_compare import (
    Ernie4_5CompareForCausalLM,
)
from dmi_vllm_integration.models.ernie45 import (
    Ernie4_5PForCausalLM,
    _apply_ernie45_attention_contract,
)
from tests.oracles.llama_compare import LlamaCompareForCausalLM
from dmi_vllm_integration.models.llama import LlamaPForCausalLM


pytestmark = pytest.mark.vllm


def test_ernie45_preserves_upstream_llama_class_contracts() -> None:
    assert ARCHITECTURE_REMAP["Ernie4_5ForCausalLM"] == "DMIErnie4_5ForCausalLM"
    assert issubclass(Ernie4_5ForCausalLM, nn.Module)
    assert issubclass(Ernie4_5PForCausalLM, LlamaPForCausalLM)
    assert issubclass(Ernie4_5CompareForCausalLM, LlamaCompareForCausalLM)
    assert (
        Ernie4_5PForCausalLM.packed_modules_mapping
        == Ernie4_5ForCausalLM.packed_modules_mapping
    )
    assert (
        vars(Ernie4_5PForCausalLM.hf_to_vllm_mapper)
        == vars(Ernie4_5ForCausalLM.hf_to_vllm_mapper)
    )
    assert (
        Ernie4_5PForCausalLM.embedding_modules
        == Ernie4_5ForCausalLM.embedding_modules
    )


def test_ernie45_replays_non_neox_rope_and_bias_free_output_projection() -> None:
    rotary = SimpleNamespace(is_neox_style=True)
    output_projection = SimpleNamespace(
        bias=torch.tensor([1.0]),
        skip_bias_add=False,
    )
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    rotary_emb=rotary,
                    o_proj=output_projection,
                )
            )
        ]
    )

    _apply_ernie45_attention_contract(model)

    assert rotary.is_neox_style is False
    assert output_projection.bias is None
    assert output_projection.skip_bias_add is True


def test_ernie45_model_wide_manifest_has_203_truthful_families() -> None:
    subject = Ernie4_5PForCausalLM.__new__(Ernie4_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.model = SimpleNamespace(layers=[None] * 18)
    specs = subject.get_hook_specs(model_wide=True)
    layer_types = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_Q,
        HOOK_TYPE_K,
        HOOK_TYPE_V,
        HOOK_TYPE_Z,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_OUT,
    ]

    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 18),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert len(specs) == 203
    assert all(spec.module is None for spec in specs)


def test_ernie45_global_hook_points_keep_llama_dtypes_and_token_axes() -> None:
    subject = Ernie4_5PForCausalLM.__new__(Ernie4_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=0,
        layers=[],
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()

    specs = subject.get_hook_specs()

    assert specs[0].dtype == torch.int32
    assert specs[0].dim0_is_actual_tokens is True
    assert specs[1].dim0_is_actual_tokens is True
    assert specs[-3].dim0_is_actual_tokens is True
    assert specs[-2].dim0_is_actual_tokens is True
    assert specs[-1].dim0_is_actual_tokens is False


def test_inherited_llama_compare_uses_head_dtype_for_logits(monkeypatch) -> None:
    subject = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=100,
        ),
        model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            head_dtype=torch.float32,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        allocations.append((shape, kwargs))
        return shape

    monkeypatch.setattr(llama_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    llama_compare.LlamaCompareForCausalLM.allocate_compare_buffers(
        subject,
        8,
        vllm_config,
    )

    assert any(
        shape == (4, 100) and kwargs["dtype"] is torch.float32
        for shape, kwargs in allocations
    )


def test_inherited_llama_compare_skips_non_logits_pipeline_stages() -> None:
    subject = SimpleNamespace(
        logits_processor=lambda _head, _hidden: None,
        lm_head=object(),
        hook_final_logits=lambda _logits: pytest.fail(
            "non-last PP stages must not emit final-logit hooks"
        ),
        _buf_final_logits=object(),
    )

    assert LlamaCompareForCausalLM.compute_logits(subject, torch.empty(0)) is None
