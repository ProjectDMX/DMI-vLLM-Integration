"""CPU contracts for MiniMax-M2.7 monitoring support."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import dmi_vllm_integration.models.minimax_m2 as minimax_m2_p
import tests.oracles.minimax_m2_compare as minimax_m2_compare
import vllm.model_executor.models.minimax_m2 as upstream_minimax_m2
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode
from vllm.model_executor.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2MoE,
)
from tests.oracles.minimax_m2_compare import (
    MiniMaxM2CompareForCausalLM,
)
from dmi_vllm_integration.models.minimax_m2 import (
    MiniMaxM2PAttention,
    MiniMaxM2PDecoderLayer,
    MiniMaxM2PForCausalLM,
    MiniMaxM2PModel,
    MiniMaxM2PMoE,
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


def _quantization_config() -> dict:
    return {
        "activation_scheme": "dynamic",
        "fmt": "float8_e4m3fn",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "gate",
            "e_score_correction_bias",
            "lm_head",
        ],
    }


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "minimax_m2",
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rotary_dim": 64,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "shared_intermediate_size": 0,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "output_router_logits": False,
        "use_qk_norm": True,
        "qk_norm_type": "per_layer",
        "max_position_embeddings": 204_800,
        "vocab_size": 200_064,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "router_jitter_noise": 0.0,
        "num_mtp_modules": 3,
        "mtp_transformer_layers": 1,
        "use_mtp": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "attn_type_list": [1] * 62,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 5_000_000,
        },
        "quantization_config": _quantization_config(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_minimax_m27_preserves_upstream_class_and_loader_contracts() -> None:
    assert ARCHITECTURE_REMAP["MiniMaxM2ForCausalLM"] == "DMIMiniMaxM2ForCausalLM"
    assert issubclass(MiniMaxM2PForCausalLM, MiniMaxM2ForCausalLM)
    assert issubclass(MiniMaxM2CompareForCausalLM, MiniMaxM2PForCausalLM)
    assert issubclass(MiniMaxM2PModel, MiniMaxM2Model)
    assert issubclass(MiniMaxM2PDecoderLayer, MiniMaxM2DecoderLayer)
    assert issubclass(MiniMaxM2PAttention, MiniMaxM2Attention)
    assert issubclass(MiniMaxM2PMoE, MiniMaxM2MoE)
    assert MiniMaxM2PForCausalLM.load_weights is MiniMaxM2ForCausalLM.load_weights
    assert (
        MiniMaxM2PForCausalLM.packed_modules_mapping
        == MiniMaxM2ForCausalLM.packed_modules_mapping
    )
    assert vars(MiniMaxM2PForCausalLM.hf_to_vllm_mapper) == vars(
        MiniMaxM2ForCausalLM.hf_to_vllm_mapper
    )


def test_minimax_m27_constructs_dmi_backbone_before_compile_wrapper() -> None:
    params = signature(MiniMaxM2PForCausalLM.__init__).parameters

    assert params["model_type"].default is MiniMaxM2PModel


def test_minimax_m27_constructor_uses_backbone_factory() -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    def model_type(*, vllm_config, prefix):
        observed.update(vllm_config=vllm_config, prefix=prefix)
        raise ConstructionStopped

    config = _config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            max_model_len=32_768,
        ),
        quant_config=None,
    )

    with pytest.raises(ConstructionStopped):
        MiniMaxM2PForCausalLM(
            vllm_config=vllm_config,
            prefix="root",
            model_type=model_type,
        )

    assert config.max_model_len == 32_768
    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.model",
    }


def test_minimax_m27_compile_wrapper_captures_dmi_backbone_forward(
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
        upstream_minimax_m2,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_minimax_m2,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, nn.ModuleList()),
    )
    monkeypatch.setattr(
        upstream_minimax_m2,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        upstream_minimax_m2,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_minimax_m2,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    config = _config(num_hidden_layers=0)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        cache_config=SimpleNamespace(),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    model = MiniMaxM2PModel(vllm_config=vllm_config)

    assert captured == [MiniMaxM2PModel.forward]
    assert isinstance(model.hook_embed, HookPoint)
    assert isinstance(model.hook_resid_final, HookPoint)
    assert isinstance(model.hook_final_ln, HookPoint)


def test_minimax_m27_instrumenter_rejects_late_class_swapping() -> None:
    with pytest.raises(TypeError, match="constructed as its DMI class"):
        minimax_m2_p._instrument_minimax_m27_model(SimpleNamespace())


def test_minimax_m27_model_shape_reads_moe_geometry() -> None:
    shape = make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.hidden_dim == 3072
    assert shape.num_heads == 48
    assert shape.num_kv_heads == 8
    assert shape.head_dim == 128
    assert shape.num_experts == 256
    assert shape.top_k == 8


def test_minimax_m27_attention_hooks_observe_normalized_pre_rope_qkv(
    monkeypatch,
) -> None:
    subject = MiniMaxM2PAttention.__new__(MiniMaxM2PAttention)
    nn.Module.__init__(subject)
    subject.num_heads = 2
    subject.num_kv_heads = 1
    subject.head_dim = 2
    subject.q_size = 4
    subject.kv_size = 2
    subject.q_norm = object()
    subject.k_norm = object()

    class Projection(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class Rotary(nn.Module):
        def forward(self, _positions, q, k):
            return q + 100, k + 200

    class Attention(nn.Module):
        def forward(self, q, k, v):
            assert q.min() >= 100
            assert k.min() >= 200
            assert v.shape == (2, 2)
            return q - 90

    class OutputProjection(nn.Module):
        def forward(self, value):
            return value + 5, None

    q = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    k = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    v = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 20
    monkeypatch.setattr(
        minimax_m2_p.MiniMaxText01RMSNormTP,
        "forward_qkv",
        lambda *_args: (q, k, v),
    )
    subject.qkv_proj = Projection()
    subject.rotary_emb = Rotary()
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

    output = subject(torch.tensor([0]), torch.zeros(2, 8))

    assert torch.equal(captured["q"].flatten(1), q)
    assert torch.equal(captured["k"].flatten(1), k)
    assert torch.equal(captured["v"].flatten(1), v)
    assert torch.equal(captured["z"], q + 10)
    assert torch.equal(output, q + 15)


def test_minimax_m27_routing_hooks_observe_fp32_gate_values() -> None:
    subject = MiniMaxM2PMoE.__new__(MiniMaxM2PMoE)
    nn.Module.__init__(subject)
    events = []

    class Gate(nn.Module):
        def forward(self, hidden_states):
            events.append("gate")
            return (
                torch.arange(
                    hidden_states.shape[0] * 4,
                    dtype=torch.float32,
                ).reshape(hidden_states.shape[0], 4),
                None,
            )

    class Router:
        observer = None
        calls = 0

        def set_routing_observer(self, observer):
            self.observer = observer

        def select_experts(self, *, hidden_states, router_logits):
            self.calls += 1
            events.append("select")
            route = (
                torch.full((2, 2), 0.5, dtype=torch.float32),
                torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )
            if self.observer is not None:
                self.observer(router_logits, *route)
            return route

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = Router()

        def forward(self, *, hidden_states, router_logits):
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            events.append("experts")
            assert router_logits.dtype == torch.float32
            return hidden_states + 1

    subject.gate = Gate()
    subject.experts = Experts()
    captured = {}
    for name in ("router_logits", "topk_ids", "topk_weights"):
        hook = HookPoint()
        hook.register_forward_hook(
            lambda _module, _args, output, name=name: captured.setdefault(
                name,
                output.clone(),
            )
        )
        setattr(subject, f"hook_{name}", hook)
    subject.experts.router.set_routing_observer(subject._observe_routing)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["gate", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].dtype == torch.float32
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32
    assert subject.experts.router.calls == 1


def test_minimax_m27_compare_buffers_cover_every_family(monkeypatch) -> None:
    subject = MiniMaxM2CompareForCausalLM.__new__(MiniMaxM2CompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=2)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(),
                block_sparse_moe=SimpleNamespace(),
            )
            for _ in range(2)
        ],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(minimax_m2_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        minimax_m2_compare,
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
    assert buffers["q_L0"].shape == (16, 12, 128)
    assert buffers["k_L0"].shape == (16, 2, 128)
    assert buffers["v_L0"].shape == (16, 2, 128)
    assert buffers["z_L0"].shape == (16, 1536)
    assert buffers["router_logits_L0"].shape == (16, 256)
    assert buffers["router_logits_L0"].dtype == torch.float32
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 200_064)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names", "args"),
    [
        (
            MiniMaxM2PAttention,
            MiniMaxM2Attention,
            ("q", "k", "v", "z"),
            (torch.tensor([0]), torch.tensor([[1.0]])),
        ),
        (
            MiniMaxM2PMoE,
            MiniMaxM2MoE,
            ("router_logits", "topk_ids", "topk_weights"),
            (torch.tensor([[1.0]]),),
        ),
        (
            MiniMaxM2PDecoderLayer,
            MiniMaxM2DecoderLayer,
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
def test_minimax_m27_disabled_hooks_delegate_to_upstream(
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


def test_minimax_m27_model_wide_manifest_has_873_families() -> None:
    subject = MiniMaxM2PForCausalLM.__new__(MiniMaxM2PForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=62,
        layers=[None] * 62,
        hook_embed=HookPoint(),
        hook_resid_final=HookPoint(),
        hook_final_ln=HookPoint(),
    )
    per_layer = [
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

    assert len(specs) == 873
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(per_layer * 62),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
