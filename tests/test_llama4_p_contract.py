"""CPU contracts for decoder-only Llama 4 Scout monitoring support."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import dmi_vllm_integration.models.llama4 as llama4_p
import dmi_vllm_integration.models.mllama4 as mllama4_p
import tests.oracles.mllama4_compare as mllama4_compare
import vllm.model_executor.models.llama as upstream_llama
from vllm.model_executor.models.llama4 import (
    Llama4Attention,
    Llama4DecoderLayer,
    Llama4ForCausalLM,
    Llama4Model,
    Llama4MoE,
)
from dmi_vllm_integration.models.llama4 import (
    Llama4PAttention,
    Llama4PDecoderLayer,
    Llama4PForCausalLM,
    Llama4PModel,
    Llama4PMoE,
    _instrument_llama4_language_model,
    _require_supported_llama4_scout_config,
    _require_supported_llama4_scout_text_config,
)
from vllm.model_executor.models.mllama4 import Llama4ForConditionalGeneration
from tests.oracles.mllama4_compare import (
    Llama4CompareForConditionalGeneration,
)
from dmi_vllm_integration.models.mllama4 import (
    Llama4PForConditionalGeneration,
)

from monitoring.integration_api.v1 import HookPoint
from monitoring.integration_api.v1 import (
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
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
)
from dmi_vllm_integration.model_shape import make_model_shape_from_hf_config

pytestmark = pytest.mark.vllm


def _text_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "llama4_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "output_router_logits": False,
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "intermediate_size_mlp": 16_384,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_local_experts": 16,
        "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 1,
        "use_qk_norm": True,
        "no_rope_layers": [1, 1, 1, 0] * 12,
        "attention_chunk_size": 8192,
        "max_position_embeddings": 10_485_760,
        "vocab_size": 202_048,
        "rope_theta": None,
        "rope_parameters": {
            "rope_type": "llama3",
            "factor": 16.0,
            "original_max_position_embeddings": 8192,
            "high_freq_factor": 1.0,
            "low_freq_factor": 1.0,
            "rope_theta": 500_000.0,
        },
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _outer_config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "llama4",
        "text_config": _text_config(),
        "vision_config": SimpleNamespace(
            model_type="llama4_vision_model",
            hidden_size=1408,
            num_hidden_layers=34,
            image_size=336,
            patch_size=14,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _parallel(**overrides) -> SimpleNamespace:
    values = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_sequence_parallel_moe": False,
        "enable_eplb": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_llama4_preserves_public_wrapper_and_loader_contracts() -> None:
    assert (
        ARCHITECTURE_REMAP["Llama4ForConditionalGeneration"]
        == "DMILlama4ForConditionalGeneration"
    )
    assert issubclass(
        Llama4PForConditionalGeneration,
        Llama4ForConditionalGeneration,
    )
    assert issubclass(
        Llama4CompareForConditionalGeneration,
        Llama4PForConditionalGeneration,
    )
    assert issubclass(Llama4PForCausalLM, Llama4ForCausalLM)
    assert issubclass(Llama4PModel, Llama4Model)
    assert issubclass(Llama4PDecoderLayer, Llama4DecoderLayer)
    assert issubclass(Llama4PAttention, Llama4Attention)
    assert issubclass(Llama4PMoE, Llama4MoE)
    assert (
        Llama4PForConditionalGeneration.load_weights
        is Llama4ForConditionalGeneration.load_weights
    )
    assert (
        Llama4PForConditionalGeneration.get_mm_mapping
        is Llama4ForConditionalGeneration.get_mm_mapping
    )
    assert (
        Llama4PForConditionalGeneration.packed_modules_mapping
        == Llama4ForConditionalGeneration.packed_modules_mapping
    )


def test_llama4_text_factory_constructs_dmi_backbone() -> None:
    observed = {}

    class Model:
        def __init__(self, *, vllm_config, prefix, layer_type):
            observed.update(
                vllm_config=vllm_config,
                prefix=prefix,
                layer_type=layer_type,
            )

    original = llama4_p.Llama4PModel
    llama4_p.Llama4PModel = Model
    try:
        vllm_config = object()
        result = Llama4PForCausalLM._init_model(
            object(),
            vllm_config,
            prefix="root.model",
            layer_type=Llama4PDecoderLayer,
        )
    finally:
        llama4_p.Llama4PModel = original

    assert isinstance(result, Model)
    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.model",
        "layer_type": Llama4PDecoderLayer,
    }


def test_llama4_compile_wrappers_capture_dmi_backbone_forward(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(compilation_counter, "num_models_seen", 0)
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "__init__",
        lambda self, **_kwargs: captured.append(self.forward.__func__),
    )
    monkeypatch.setattr(
        upstream_llama,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        upstream_llama,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_llama,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, nn.ModuleList()),
    )
    monkeypatch.setattr(
        upstream_llama,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        upstream_llama,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    config = _text_config(num_hidden_layers=0)
    config.rms_norm_eps = 1e-5
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    Llama4PModel(vllm_config=vllm_config)

    assert captured
    assert set(captured) == {Llama4PModel.forward}


def test_llama4_instrumenter_rejects_late_class_swapping() -> None:
    upstream_language_model = SimpleNamespace(config=_text_config())
    with pytest.raises(TypeError, match="language model must be constructed"):
        _instrument_llama4_language_model(upstream_language_model, _parallel())

    language_model = Llama4PForCausalLM.__new__(Llama4PForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = _text_config()
    language_model.model = nn.Identity()
    with pytest.raises(TypeError, match="backbone must be constructed"):
        _instrument_llama4_language_model(language_model, _parallel())


def test_llama4_multimodal_constructor_uses_dmi_language_model_factory(
    monkeypatch,
) -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    class LanguageModel(nn.Module):
        pass

    inner_vllm_config = object()

    def with_hf_config(hf_config, architectures):
        observed.update(hf_config=hf_config, architectures=architectures)
        return inner_vllm_config

    def initialize_model(*, vllm_config, prefix, model_class):
        observed.update(
            vllm_config=vllm_config,
            prefix=prefix,
            model_class=model_class,
        )
        raise ConstructionStopped

    config = _outer_config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            multimodal_config=SimpleNamespace(mm_encoder_tp_mode="weights"),
            dtype=torch.bfloat16,
        ),
        parallel_config=_parallel(),
        quant_config=None,
        with_hf_config=with_hf_config,
    )
    monkeypatch.setattr(mllama4_p, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(
        mllama4_p,
        "Llama4VisionModel",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        mllama4_p,
        "Llama4MultiModalProjector",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(mllama4_p, "initialize_model", initialize_model)
    monkeypatch.setattr(
        Llama4PForConditionalGeneration,
        "_mark_tower_model",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        Llama4PForConditionalGeneration,
        "_mark_language_model",
        lambda *_args, **_kwargs: nullcontext(),
    )

    with pytest.raises(ConstructionStopped):
        Llama4PForConditionalGeneration(
            vllm_config=vllm_config,
            prefix="root",
            language_model_type=LanguageModel,
        )

    assert observed == {
        "hf_config": config.text_config,
        "architectures": ["LlamaForCausalLM"],
        "vllm_config": inner_vllm_config,
        "prefix": "root.language_model",
        "model_class": LanguageModel,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"interleave_moe_layer_step": 0},
        {"interleave_moe_layer_step": 2},
    ],
)
def test_llama4_requires_moe_in_every_decoder_layer(overrides) -> None:
    with pytest.raises(NotImplementedError, match="MoE in every decoder layer"):
        _require_supported_llama4_scout_text_config(
            _text_config(**overrides),
            _parallel(),
        )


@pytest.mark.parametrize(
    "config_overrides,parallel_overrides,dtype",
    [
        ({"hidden_size": 4096}, {}, None),
        ({"no_rope_layers": [0] * 48}, {}, None),
        ({"rope_parameters": {"rope_type": "yarn"}}, {}, None),
        ({"quantization_config": {"quant_method": "fp8"}}, {}, None),
        ({}, {"tensor_parallel_size": 1}, None),
        ({}, {"pipeline_parallel_size": 2}, None),
        ({}, {"enable_expert_parallel": True}, None),
        ({}, {"use_sequence_parallel_moe": True}, None),
        ({}, {"enable_eplb": True}, None),
        ({}, {}, torch.float16),
    ],
)
def test_llama4_does_not_reject_upstream_owned_or_shared_runtime_options(
    config_overrides,
    parallel_overrides,
    dtype,
) -> None:
    _require_supported_llama4_scout_text_config(
        _text_config(**config_overrides),
        _parallel(**parallel_overrides),
        dtype=dtype,
    )


def test_llama4_outer_validator_ignores_upstream_owned_vision_fields() -> None:
    config = _outer_config(vision_config=None)
    _require_supported_llama4_scout_config(config, _parallel())


def test_llama4_model_shape_uses_the_nested_text_config() -> None:
    shape = make_model_shape_from_hf_config(_outer_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 5120
    assert shape.num_heads == 40
    assert shape.num_kv_heads == 8
    assert shape.head_dim == 128
    assert shape.num_experts == 16
    assert shape.top_k == 1


def test_llama4_moe_hooks_observe_custom_router_outputs() -> None:
    subject = Llama4PMoE.__new__(Llama4PMoE)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    events = []

    class Router(nn.Module):
        def forward(self, hidden_states):
            events.append("router")
            return hidden_states + 10, None

    class ExpertRouter:
        observer = None
        calls = 0

        def set_routing_observer(self, observer):
            self.observer = observer

        def select_experts(self, *, hidden_states, router_logits):
            self.calls += 1
            events.append("select")
            assert router_logits.shape == (2, 4)
            route = (
                torch.sigmoid(router_logits[:, :1].float()),
                torch.zeros((2, 1), dtype=torch.int32),
            )
            if self.observer is not None:
                self.observer(router_logits, *route)
            return route

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ExpertRouter()

        def forward(self, *, hidden_states, router_logits):
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            events.append("experts")
            return hidden_states + 1

    subject.router = Router()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output,
            )
        )
        setattr(subject, f"hook_{name}", hook)
    subject.experts.router.set_routing_observer(subject._observe_routing)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["router", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].shape == (2, 4)
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32
    assert subject.experts.router.calls == 1


def test_llama4_multimodal_wrapper_delegates_only_decoder_manifest() -> None:
    subject = Llama4PForConditionalGeneration.__new__(Llama4PForConditionalGeneration)
    nn.Module.__init__(subject)
    sentinel = [object()]
    calls = []
    subject.language_model = SimpleNamespace(
        get_hook_specs=lambda **kwargs: calls.append(kwargs) or sentinel
    )
    subject.vision_model = SimpleNamespace()
    subject.multi_modal_projector = SimpleNamespace()

    result = subject.get_hook_specs(model_wide=True)

    assert result is sentinel
    assert calls == [{"model_wide": True}]
    assert not hasattr(subject.vision_model, "get_hook_specs")
    assert not hasattr(subject.multi_modal_projector, "get_hook_specs")


def test_llama4_compare_buffers_cover_every_decoder_family(monkeypatch) -> None:
    subject = Llama4CompareForConditionalGeneration.__new__(
        Llama4CompareForConditionalGeneration
    )
    nn.Module.__init__(subject)
    config = _text_config(num_hidden_layers=2)
    language_model = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            start_layer=0,
            end_layer=2,
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(),
                    feed_forward=SimpleNamespace(),
                )
                for _ in range(2)
            ],
        ),
    )
    subject.language_model = language_model
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(mllama4_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        mllama4_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 33
    assert len(buffers) == 33
    assert buffers["q_L0"].shape == (16, 10, 128)
    assert buffers["k_L0"].shape == (16, 2, 128)
    assert buffers["router_logits_L0"].shape == (16, 16)
    assert buffers["topk_ids_L0"].shape == (16, 1)
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 202_048)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args"),
    [
        (
            Llama4PAttention,
            Llama4Attention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
        ),
        (
            Llama4PMoE,
            Llama4MoE,
            ("router_logits", "topk_ids", "topk_weights"),
            (torch.tensor([[1.0]]),),
        ),
        (
            Llama4PDecoderLayer,
            Llama4DecoderLayer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
            (torch.tensor([0]), torch.tensor([[1.0]]), None),
        ),
    ],
)
def test_llama4_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
    args,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *forward_args):
        observed.append(forward_args)
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)

    result = subject(*args)

    assert result is marker
    assert observed == [args]


def test_llama4_model_wide_manifest_has_677_decoder_families() -> None:
    subject = Llama4PForCausalLM.__new__(Llama4PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _text_config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=48,
        layers=[None] * 48,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
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
        HOOK_TYPE_ROUTER_LOGITS,
        HOOK_TYPE_TOPK_IDS,
        HOOK_TYPE_TOPK_WEIGHTS,
        HOOK_TYPE_MLP_OUT,
    ]

    specs = subject.get_hook_specs(model_wide=True)

    assert len(specs) == 677
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 48),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
    assert specs[13].dtype == torch.int32
    assert specs[14].dtype == torch.float32
