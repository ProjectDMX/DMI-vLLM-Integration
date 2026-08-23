"""CPU contracts for decoder-only Qwen3.6-27B monitoring support."""

from __future__ import annotations

from contextlib import nullcontext
from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import dmi_vllm_integration.models.qwen3_5 as qwen3_5_p
import tests.oracles.qwen3_5_compare as qwen3_5_compare
import vllm.model_executor.models.qwen3_5 as upstream_qwen3_5
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from tests.oracles.qwen3_5_compare import (
    Qwen3_5CompareForConditionalGeneration,
)
from dmi_vllm_integration.models.qwen3_5 import (
    Qwen3_5PAttention,
    Qwen3_5PDecoderLayer,
    Qwen3_5PForCausalLM,
    Qwen3_5PForConditionalGeneration,
    Qwen3_5PMLP,
    Qwen3_5PModel,
    _instrument_qwen36_language_model,
    _require_supported_qwen36_config,
    _require_supported_qwen36_text_config,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
)

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
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from dmi_vllm_integration.model_shape import make_model_shape_from_hf_config

pytestmark = pytest.mark.vllm


def _layer_types(count: int = 64) -> list[str]:
    return [
        "full_attention" if (layer_no + 1) % 4 == 0 else "linear_attention"
        for layer_no in range(count)
    ]


def _text_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "qwen3_5_text",
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17_408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "max_position_embeddings": 262_144,
        "vocab_size": 248_320,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "qkv_bias": False,
        "tie_word_embeddings": False,
        "layer_scale": None,
        "layer_types": _layer_types(),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _outer_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "qwen3_5",
        "language_model_only": False,
        "text_config": _text_config(),
        "vision_config": SimpleNamespace(
            model_type="qwen3_5",
            hidden_size=1152,
            intermediate_size=4304,
            hidden_act="gelu_pytorch_tanh",
            depth=27,
            num_heads=16,
            in_channels=3,
            num_position_embeddings=2304,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=5120,
            deepstack_visual_indexes=[],
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen36_preserves_public_wrapper_and_loader_contracts() -> None:
    assert (
        ARCHITECTURE_REMAP["Qwen3_5ForConditionalGeneration"]
        == "DMIQwen3_5ForConditionalGeneration"
    )
    assert issubclass(
        Qwen3_5PForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
    )
    assert issubclass(
        Qwen3_5CompareForConditionalGeneration,
        Qwen3_5PForConditionalGeneration,
    )
    assert issubclass(Qwen3_5PForCausalLM, Qwen3_5ForCausalLM)
    assert issubclass(Qwen3_5PModel, Qwen3_5Model)
    assert issubclass(Qwen3_5PDecoderLayer, Qwen3_5DecoderLayer)
    assert issubclass(Qwen3_5PAttention, Qwen3NextAttention)
    assert issubclass(Qwen3_5PMLP, Qwen3NextMLP)
    assert (
        Qwen3_5PForConditionalGeneration.load_weights
        is Qwen3_5ForConditionalGeneration.load_weights
    )
    assert (
        Qwen3_5PForConditionalGeneration.get_mm_mapping
        is Qwen3_5ForConditionalGeneration.get_mm_mapping
    )
    assert (
        Qwen3_5PForConditionalGeneration.packed_modules_mapping
        == Qwen3_5ForConditionalGeneration.packed_modules_mapping
    )


def test_qwen36_constructs_dmi_classes_before_compile_wrapper() -> None:
    causal_params = signature(Qwen3_5PForCausalLM.__init__).parameters
    outer_params = signature(
        Qwen3_5PForConditionalGeneration.__init__
    ).parameters

    assert causal_params["model_type"].default is Qwen3_5PModel
    assert (
        outer_params["language_model_type"].default
        is Qwen3_5PForCausalLM
    )


def test_qwen36_causal_constructor_uses_backbone_factory() -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    def model_type(*, vllm_config, prefix):
        observed.update(vllm_config=vllm_config, prefix=prefix)
        raise ConstructionStopped

    config = _text_config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config,
            dtype=torch.bfloat16,
        ),
        parallel_config=_parallel(),
        quant_config=None,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(),
    )

    with pytest.raises(ConstructionStopped):
        Qwen3_5PForCausalLM(
            vllm_config=vllm_config,
            prefix="root",
            model_type=model_type,
        )

    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.model",
    }


def test_qwen36_outer_constructor_uses_language_model_factory(
    monkeypatch,
) -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    def language_model_type(*, vllm_config, prefix):
        observed.update(vllm_config=vllm_config, prefix=prefix)
        raise ConstructionStopped

    multimodal_config = SimpleNamespace(
        mm_encoder_tp_mode="tensor",
        is_multimodal_pruning_enabled=lambda: False,
        video_pruning_rate=0.0,
    )
    config = _outer_config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            hf_text_config=config.text_config,
            multimodal_config=multimodal_config,
            dtype=torch.bfloat16,
        ),
        parallel_config=_parallel(),
        quant_config=None,
    )
    monkeypatch.setattr(
        qwen3_5_p,
        "cached_tokenizer_from_config",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        qwen3_5_p,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        Qwen3_5PForConditionalGeneration,
        "_mark_tower_model",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        Qwen3_5PForConditionalGeneration,
        "_mark_language_model",
        lambda *_args, **_kwargs: nullcontext(),
    )

    with pytest.raises(ConstructionStopped):
        Qwen3_5PForConditionalGeneration(
            vllm_config=vllm_config,
            prefix="root",
            language_model_type=language_model_type,
        )

    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.language_model",
    }


def test_qwen36_compile_wrapper_captures_dmi_backbone_forward(
    monkeypatch,
) -> None:
    captured = []
    monkeypatch.setattr(compilation_counter, "num_models_seen", 0)
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "__init__",
        lambda self, **_kwargs: captured.append(self.forward.__func__),
    )
    monkeypatch.setattr(
        upstream_qwen3_5,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_qwen3_5,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, nn.ModuleList()),
    )
    monkeypatch.setattr(
        upstream_qwen3_5,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        upstream_qwen3_5,
        "Qwen3_5RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_qwen3_5,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    config = _text_config(num_hidden_layers=0, layer_types=[])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    Qwen3_5PModel(vllm_config=vllm_config)

    assert captured == [Qwen3_5PModel.forward]


def test_qwen36_instrumenter_rejects_late_class_swapping() -> None:
    upstream_language_model = SimpleNamespace(config=_text_config())

    with pytest.raises(TypeError, match="constructed as its DMI class"):
        _instrument_qwen36_language_model(upstream_language_model, _parallel())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model_type": "qwen3_5_moe_text"}, "dense text variant"),
        ({"layer_scale": 1.0}, "layer scaling"),
    ],
)
def test_qwen36_rejects_unimplemented_text_branches(
    overrides,
    match,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _require_supported_qwen36_text_config(
            _text_config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "config_overrides,parallel_overrides,dtype",
    [
        ({"hidden_size": 4096}, {}, None),
        ({"layer_types": ["full_attention"] * 64}, {}, None),
        ({"rope_parameters": None}, {}, None),
        ({"quantization_config": {"quant_method": "fp8"}}, {}, None),
        ({}, {"tensor_parallel_size": 2}, None),
        ({}, {"pipeline_parallel_size": 2}, None),
        ({}, {"enable_expert_parallel": True}, None),
        ({}, {"use_sequence_parallel_moe": True}, None),
        ({}, {}, torch.float16),
    ],
)
def test_qwen36_does_not_reject_upstream_owned_or_shared_runtime_options(
    config_overrides,
    parallel_overrides,
    dtype,
) -> None:
    _require_supported_qwen36_text_config(
        _text_config(**config_overrides),
        _parallel(**parallel_overrides),
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "outer_overrides",
    [
        {"model_type": "qwen3_vl"},
        {"language_model_only": True},
        {"vision_config": None},
    ],
)
def test_qwen36_outer_validator_ignores_upstream_owned_wrapper_fields(
    outer_overrides,
) -> None:
    _require_supported_qwen36_config(
        _outer_config(**outer_overrides),
        _parallel(),
    )


def test_qwen36_model_shape_uses_the_nested_text_config() -> None:
    shape = make_model_shape_from_hf_config(_outer_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 5120
    assert shape.intermediate_dim == 17_408
    assert shape.num_heads == 24
    assert shape.num_kv_heads == 4
    assert shape.head_dim == 256
    assert shape.num_experts == 0
    assert shape.top_k == 0


def test_qwen36_mlp_post_hook_observes_post_activation_values() -> None:
    subject = Qwen3_5PMLP.__new__(Qwen3_5PMLP)
    nn.Module.__init__(subject)

    class GateUp(nn.Module):
        def forward(self, value):
            return value + 2, None

    class Activation(nn.Module):
        def forward(self, value):
            return value * 3

    class Down(nn.Module):
        def forward(self, value):
            return value - 5, None

    subject.gate_up_proj = GateUp()
    subject.act_fn = Activation()
    subject.down_proj = Down()
    subject.expert_gate = None
    subject.hook_post = HookPoint()
    captured = []
    subject.hook_post.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    inputs = torch.tensor([[1.0, 2.0]])

    output = subject(inputs)

    expected_post = (inputs + 2) * 3
    assert torch.equal(captured[0], expected_post)
    assert torch.equal(output, expected_post - 5)


def test_qwen36_full_attention_hooks_observe_gated_decoder_values() -> None:
    subject = Qwen3_5PAttention.__new__(Qwen3_5PAttention)
    nn.Module.__init__(subject)
    subject.num_heads = 2
    subject.num_kv_heads = 1
    subject.head_dim = 2

    class Projection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class Attention(nn.Module):
        def forward(self, q, k, v):
            assert k.shape == v.shape == (2, 2)
            return q + 1

    class OutputProjection(nn.Module):
        def forward(self, value):
            return value + 10, None

    subject.qkv_proj = Projection()
    subject._project_qkv_gate = lambda _qkv, _positions: (
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(4, dtype=torch.float32).reshape(2, 2) + 20,
        torch.zeros(2, 4),
    )
    subject.attn = Attention()
    subject.o_proj = OutputProjection()
    captured = {}
    for name in ("q", "k", "v", "z"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output.clone(),
            )
        )
        setattr(subject, f"hook_{name}", hook)

    output = subject(torch.tensor([0]), torch.zeros(2, 4))

    expected_z = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1) * 0.5
    assert captured["q"].shape == (2, 2, 2)
    assert captured["k"].shape == captured["v"].shape == (2, 1, 2)
    assert torch.equal(captured["z"], expected_z)
    assert torch.equal(output, expected_z + 10)


def test_qwen36_multimodal_wrapper_delegates_only_decoder_manifest() -> None:
    subject = Qwen3_5PForConditionalGeneration.__new__(Qwen3_5PForConditionalGeneration)
    nn.Module.__init__(subject)
    sentinel = [object()]
    calls = []
    subject.language_model = SimpleNamespace(
        get_hook_specs=lambda **kwargs: calls.append(kwargs) or sentinel
    )
    subject.visual = SimpleNamespace()

    result = subject.get_hook_specs(model_wide=True)

    assert result is sentinel
    assert calls == [{"model_wide": True}]
    assert not hasattr(subject.visual, "get_hook_specs")


def test_qwen36_outer_forward_captures_tokens_before_decoder_bypass(
    monkeypatch,
) -> None:
    subject = Qwen3_5PForConditionalGeneration.__new__(Qwen3_5PForConditionalGeneration)
    nn.Module.__init__(subject)
    hook = HookPoint()
    captured = []
    hook.register_forward_hook(
        lambda _module, _args, output: captured.append(output.clone())
    )
    subject.language_model = SimpleNamespace(hook_token_ids=hook)
    monkeypatch.setattr(
        qwen3_5_p,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    marker = torch.tensor([[19.0]])
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration,
        "forward",
        lambda _self, *args, **kwargs: marker,
    )
    input_ids = torch.tensor([1, 2], dtype=torch.int32)

    output = subject(input_ids, torch.tensor([0, 1]))

    assert output is marker
    assert torch.equal(captured[0], input_ids)


def test_qwen36_compare_buffers_cover_heterogeneous_decoder(monkeypatch) -> None:
    subject = Qwen3_5CompareForConditionalGeneration.__new__(
        Qwen3_5CompareForConditionalGeneration
    )
    nn.Module.__init__(subject)
    config = _text_config(num_hidden_layers=4, layer_types=_layer_types(4))
    layers = []
    for layer_type in config.layer_types:
        layer = SimpleNamespace(layer_type=layer_type, mlp=SimpleNamespace())
        if layer_type == "full_attention":
            layer.self_attn = SimpleNamespace()
        layers.append(layer)
    subject.language_model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(start_layer=0, end_layer=4, layers=layers),
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(qwen3_5_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        qwen3_5_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 41
    assert len(buffers) == 41
    assert "q_L0" not in buffers
    assert buffers["q_L3"].shape == (16, 24, 256)
    assert buffers["k_L3"].shape == (16, 4, 256)
    assert buffers["v_L3"].shape == (16, 4, 256)
    assert buffers["z_L3"].shape == (16, 6144)
    assert buffers["mlp_post_L0"].shape == (16, 17_408)
    assert buffers["final_logits"].shape == (4, 248_320)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args", "kwargs"),
    [
        (
            Qwen3_5PAttention,
            Qwen3NextAttention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
            {},
        ),
        (
            Qwen3_5PMLP,
            Qwen3NextMLP,
            ("post",),
            (torch.tensor([[1.0]]),),
            {},
        ),
        (
            Qwen3_5PDecoderLayer,
            Qwen3_5DecoderLayer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
            (torch.tensor([[1.0]]), None),
            {"positions": torch.tensor([0])},
        ),
    ],
)
def test_qwen36_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
    kwargs,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *forward_args, **forward_kwargs):
        observed.append((forward_args, forward_kwargs))
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)

    result = subject(*args, **kwargs)

    assert result is marker
    assert observed == [(args, kwargs)]


def test_qwen36_model_wide_manifest_has_581_truthful_families() -> None:
    subject = Qwen3_5PForCausalLM.__new__(Qwen3_5PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _text_config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=64,
        layers=[None] * 64,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    common = [
        HOOK_TYPE_RESID_PRE,
        HOOK_TYPE_LN1,
        HOOK_TYPE_ATTN_OUT,
        HOOK_TYPE_RESID_MID,
        HOOK_TYPE_LN2,
        HOOK_TYPE_MLP_IN,
        HOOK_TYPE_MLP_POST,
        HOOK_TYPE_MLP_OUT,
    ]
    full_only = [HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V, HOOK_TYPE_Z]
    expected = [HOOK_TYPE_TOKEN_IDS, HOOK_TYPE_EMBED]
    for layer_no, layer_type in enumerate(subject.config.layer_types):
        layer = common[:2]
        if layer_type == "full_attention":
            layer += full_only
        layer += common[2:]
        expected.extend(layer)
    expected.extend([HOOK_TYPE_RESID_FINAL, HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS])

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 581
    assert [spec.hook_type for spec in specs] == expected
    assert all(spec.module is None for spec in specs)
    assert sum(spec.hook_type == HOOK_TYPE_Q for spec in specs) == 16
    assert sum(spec.hook_type == HOOK_TYPE_MLP_POST for spec in specs) == 64
