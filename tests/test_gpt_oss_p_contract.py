"""CPU contracts for GPT-OSS monitoring support."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP
import dmi_vllm_integration.models.gpt_oss as gpt_oss_p
import tests.oracles.gpt_oss_compare as gpt_oss_compare
import vllm.model_executor.models.gpt_oss as upstream_gpt_oss
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    GptOssModel,
    MLPBlock,
    OAIAttention,
    TransformerBlock,
)
from tests.oracles.gpt_oss_compare import (
    GptOssCompareForCausalLM,
)
from dmi_vllm_integration.models.gpt_oss import (
    GptOssPAttention,
    GptOssPForCausalLM,
    GptOssPMLP,
    GptOssPModel,
    GptOssPTransformerBlock,
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


def _config(**overrides) -> SimpleNamespace:
    values = {
        "model_type": "gpt_oss",
        "hidden_act": "silu",
        "attention_bias": True,
        "tie_word_embeddings": False,
        "output_router_logits": False,
        "swiglu_limit": 7,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "experts_per_token": 4,
        "sliding_window": 128,
        "max_position_embeddings": 131_072,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "vocab_size": 201_088,
        "rope_theta": 150_000,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 150_000,
            "factor": 32,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "truncate": False,
        },
        "quantization_config": {"quant_method": "mxfp4"},
    }
    values.update(overrides)
    if "layer_types" not in overrides:
        values["layer_types"] = [
            "sliding_attention" if layer_no % 2 == 0 else "full_attention"
            for layer_no in range(values["num_hidden_layers"])
        ]
    return SimpleNamespace(**values)


def test_gpt_oss_preserves_upstream_class_and_loader_contracts() -> None:
    assert ARCHITECTURE_REMAP["GptOssForCausalLM"] == "DMIGptOssForCausalLM"
    assert issubclass(GptOssPForCausalLM, GptOssForCausalLM)
    assert issubclass(GptOssCompareForCausalLM, GptOssPForCausalLM)
    assert issubclass(GptOssPModel, GptOssModel)
    assert issubclass(GptOssPTransformerBlock, TransformerBlock)
    assert issubclass(GptOssPAttention, OAIAttention)
    assert issubclass(GptOssPMLP, MLPBlock)
    assert (
        GptOssPForCausalLM.packed_modules_mapping
        == GptOssForCausalLM.packed_modules_mapping
    )
    assert vars(GptOssPForCausalLM.hf_to_vllm_mapper) == vars(
        GptOssForCausalLM.hf_to_vllm_mapper
    )
    assert GptOssPForCausalLM.is_3d_moe_weight is True


def test_gpt_oss_constructs_dmi_backbone_before_compile_wrapper() -> None:
    params = signature(GptOssPForCausalLM.__init__).parameters

    assert params["model_type"].default is GptOssPModel


def test_gpt_oss_constructor_uses_backbone_factory() -> None:
    observed = {}

    class ConstructionStopped(Exception):
        pass

    def model_type(*, vllm_config, prefix):
        observed.update(vllm_config=vllm_config, prefix=prefix)
        raise ConstructionStopped

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_config()),
    )

    with pytest.raises(ConstructionStopped):
        GptOssPForCausalLM(
            vllm_config=vllm_config,
            prefix="root",
            model_type=model_type,
        )

    assert observed == {
        "vllm_config": vllm_config,
        "prefix": "root.model",
    }


def test_gpt_oss_compile_wrapper_captures_dmi_backbone_forward(
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
        upstream_gpt_oss,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_gpt_oss,
        "make_layers",
        lambda *_args, **_kwargs: (0, 0, nn.ModuleList()),
    )
    monkeypatch.setattr(
        upstream_gpt_oss,
        "RMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        upstream_gpt_oss,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    config = _config(num_hidden_layers=0, layer_types=[])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(mode=CompilationMode.VLLM_COMPILE),
    )

    GptOssPModel(vllm_config=vllm_config)

    assert captured == [GptOssPModel.forward]


def test_gpt_oss_instrumenter_rejects_late_class_swapping() -> None:
    upstream_model = GptOssModel.__new__(GptOssModel)
    nn.Module.__init__(upstream_model)

    with pytest.raises(TypeError, match="constructed as its DMI class"):
        gpt_oss_p._instrument_gpt_oss_model(upstream_model)


def test_gpt_oss_model_shape_reads_local_expert_geometry() -> None:
    shape = make_model_shape_from_hf_config(_config(), torch.bfloat16)

    assert shape is not None
    assert shape.num_experts == 32
    assert shape.top_k == 4
    assert shape.head_dim == 64


def test_gpt_oss_routing_hooks_observe_authoritative_router_inputs(
    monkeypatch,
) -> None:
    subject = GptOssPMLP.__new__(GptOssPMLP)
    nn.Module.__init__(subject)
    subject.is_sequence_parallel = False
    subject.hidden_size = 4
    events = []

    class Router(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append("router")
            return torch.arange(
                hidden_states.shape[0] * 4,
                dtype=torch.float32,
            ).reshape(hidden_states.shape[0], 4)

    class ExpertRouter:
        observer = None
        calls = 0

        def set_routing_observer(self, observer):
            self.observer = observer

        def select_experts(self, *, hidden_states, router_logits):
            self.calls += 1
            events.append("select")
            assert hidden_states.shape == (2, 4)
            assert router_logits.shape == (2, 4)
            route = (
                torch.full((2, 2), 0.5, dtype=torch.float32),
                torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            )
            if self.observer is not None:
                self.observer(router_logits, *route)
            return route

    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = ExpertRouter()

        def forward(self, *, hidden_states, router_logits):
            self.used_route = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            events.append("experts")
            assert router_logits.shape == (2, 4)
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
    monkeypatch.setattr(
        gpt_oss_p,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False),
    )
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = subject(hidden_states)

    assert events == ["router", "select", "experts"]
    assert torch.equal(output, hidden_states + 1)
    assert captured["router_logits"].shape == (2, 4)
    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_weights"].dtype == torch.float32
    assert subject.experts.router.calls == 1


def test_gpt_oss_compare_buffers_cover_every_manifest_family(monkeypatch) -> None:
    subject = GptOssCompareForCausalLM.__new__(GptOssCompareForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config(num_hidden_layers=2)
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[
            SimpleNamespace(attn=SimpleNamespace(), mlp=SimpleNamespace())
            for _ in range(2)
        ],
    )
    allocations = []

    def fake_empty(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, **kwargs)
        allocations.append(value)
        return value

    monkeypatch.setattr(gpt_oss_compare.torch, "empty", fake_empty)
    monkeypatch.setattr(
        gpt_oss_compare,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )

    subject.allocate_compare_buffers(16, vllm_config)
    buffers = subject.get_ref_buffers()

    assert len(allocations) == 33
    assert len(buffers) == 33
    assert buffers["q_L0"].shape == (16, 32, 64)
    assert buffers["k_L0"].shape == (16, 4, 64)
    assert buffers["router_logits_L0"].shape == (16, 32)
    assert buffers["topk_ids_L0"].dtype == torch.int32
    assert buffers["topk_weights_L0"].dtype == torch.float32
    assert buffers["final_logits"].shape == (4, 201_088)


@pytest.mark.parametrize(
    ("subject_cls", "upstream_cls", "hook_names"),
    [
        (GptOssPAttention, OAIAttention, ("q", "k", "v", "z")),
        (
            GptOssPMLP,
            MLPBlock,
            ("router_logits", "topk_ids", "topk_weights"),
        ),
        (
            GptOssPTransformerBlock,
            TransformerBlock,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
        ),
    ],
)
def test_gpt_oss_disabled_hooks_delegate_to_upstream(
    monkeypatch,
    subject_cls,
    upstream_cls,
    hook_names,
) -> None:
    subject = subject_cls.__new__(subject_cls)
    nn.Module.__init__(subject)
    for name in hook_names:
        hook = HookPoint()
        hook.enabled = False
        setattr(subject, f"hook_{name}", hook)
    marker = torch.tensor([[17.0]])
    observed = []

    def upstream_forward(_self, *args):
        observed.append(args)
        return marker

    monkeypatch.setattr(upstream_cls, "forward", upstream_forward)
    args = (
        (torch.tensor([[1.0]]), torch.tensor([0]))
        if subject_cls is GptOssPAttention
        else (
            (torch.tensor([[1.0]]),)
            if subject_cls is GptOssPMLP
            else (torch.tensor([[1.0]]), torch.tensor([0]), None)
        )
    )

    result = subject(*args)

    assert result is marker
    assert observed == [args]


def test_gpt_oss_model_wide_manifest_has_341_truthful_families() -> None:
    subject = GptOssPForCausalLM.__new__(GptOssPForCausalLM)
    nn.Module.__init__(subject)
    subject.config = _config()
    subject.hook_token_ids = HookPoint()
    subject.hook_final_logits = HookPoint()
    subject.model = SimpleNamespace(
        start_layer=0,
        end_layer=24,
        layers=[None] * 24,
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

    assert len(specs) == 341
    assert [spec.hook_type for spec in specs] == [
        HOOK_TYPE_TOKEN_IDS,
        HOOK_TYPE_EMBED,
        *(layer_types * 24),
        HOOK_TYPE_RESID_FINAL,
        HOOK_TYPE_FINAL_LN,
        HOOK_TYPE_FINAL_LOGITS,
    ]
    assert all(spec.module is None for spec in specs)
    assert specs[13].dtype == torch.int32
    assert specs[14].dtype == torch.float32
