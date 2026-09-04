"""Layer-range support in the vLLM attach path.

DMI-configurator forwards a selected layer range as
``attach_model(model, hook_selection, layers=LayerSelection(...))``. The
adaptor must (a) accept the keyword, (b) apply it to the installed local
specs the same way the HF adapter does, and (c) apply the IDENTICAL filter
to the model-wide candidate-rank spec sets, so the rank formulas and the
filtered local specs can never disagree about which layers capture.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _install_fake_vllm() -> None:
    """The adapter imports vLLM at module load; a stub keeps this suite
    runnable where vLLM is not installed (CPU CI). ``get_pp_indices`` is the
    only vllm symbol the code under test actually executes."""
    if "vllm" in sys.modules and hasattr(sys.modules["vllm"], "distributed"):
        return

    def get_pp_indices(num_layers, pp_rank, pp_size):
        start = num_layers * pp_rank // pp_size
        end = num_layers * (pp_rank + 1) // pp_size
        return start, end

    vllm = ModuleType("vllm")
    vllm.LLM = object
    distributed = ModuleType("vllm.distributed")
    ec_transfer = ModuleType("vllm.distributed.ec_transfer")
    ec_transfer.get_ec_transfer = lambda *a, **k: None
    ec_transfer.has_ec_transfer = lambda *a, **k: False
    multimodal = ModuleType("vllm.multimodal")
    multimodal.MULTIMODAL_REGISTRY = SimpleNamespace()
    v1 = ModuleType("vllm.v1")
    worker = ModuleType("vllm.v1.worker")
    gpu_worker = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = object
    compilation = ModuleType("vllm.compilation")
    cuda_graph = ModuleType("vllm.compilation.cuda_graph")

    class CUDAGraphWrapper:  # the unwrap() probe only isinstance-checks
        pass

    cuda_graph.CUDAGraphWrapper = CUDAGraphWrapper
    for name, module in (
        ("vllm", vllm),
        ("vllm.distributed", distributed),
        ("vllm.distributed.ec_transfer", ec_transfer),
        ("vllm.multimodal", multimodal),
        ("vllm.v1", v1),
        ("vllm.v1.worker", worker),
        ("vllm.v1.worker.gpu_worker", gpu_worker),
        ("vllm.compilation", compilation),
        ("vllm.compilation.cuda_graph", cuda_graph),
    ):
        sys.modules.setdefault(name, module)
    distributed.utils = ModuleType("vllm.distributed.utils")
    distributed.utils.get_pp_indices = get_pp_indices
    sys.modules["vllm.distributed.utils"] = distributed.utils
    vllm.distributed = distributed
    vllm.multimodal = multimodal
    vllm.v1 = v1
    vllm.compilation = compilation


_install_fake_vllm()

from dmi_vllm_integration.adapter import VLLMAdaptor, _VLLMHookSelection  # noqa: E402
from dmi_vllm_integration.dmi_api import (  # noqa: E402
    HOOK_TYPE_RESID_PRE,
    HookSpec,
    ModelShapeConfig,
)


def _hooks(num_layers: int = 8, hook_type: int = HOOK_TYPE_RESID_PRE):
    return tuple(
        HookSpec(hook_type, None, layer_no=layer) for layer in range(num_layers)
    )


def _cfg() -> ModelShapeConfig:
    return ModelShapeConfig(
        hidden_dim=512,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_dim=1024,
        num_experts=0,
        top_k=0,
        dtype=None,
        tp_size=1,
        tp_rank=0,
    )


def _select(hook_selection: str = "full", layers=None, num_layers: int = 8):
    return _VLLMHookSelection.from_model(
        model=SimpleNamespace(
            get_hook_specs=lambda model_wide=False: _hooks(num_layers)
        ),
        local_hooks=_hooks(num_layers),
        hook_selection=hook_selection,
        cfg=_cfg(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        hf_config=SimpleNamespace(num_hidden_layers=num_layers),
        layers=layers,
    )


class TestLayerRangeFiltersBothSides:
    def test_signature_accepts_layers(self):
        import inspect

        parameters = inspect.signature(VLLMAdaptor.attach_model).parameters
        assert "layers" in parameters, (
            "VLLMAdaptor.attach_model must accept the layers keyword "
            "DMI-configurator's attach_config() forwards"
        )

    def test_local_hooks_outside_the_range_are_filtered(self):
        from dmi.configuration.schema import LayerSelection

        selection = _select(layers=LayerSelection(2, 5))

        local_layers = {spec.layer_no for spec in selection.local_hooks}
        assert local_layers == {2, 3, 4, 5}

    def test_candidate_rank_sets_apply_the_same_filter(self):
        """The model-wide formulas must see the same layers the local specs
        do, or a per-rank byte plan disagrees with what actually captures."""
        from dmi.configuration.schema import LayerSelection

        selection = _select(layers=LayerSelection(2, 5))

        for rank_set in selection.candidate_rank_hook_sets:
            for spec in rank_set:
                assert 2 <= spec.layer_no <= 5, (
                    f"candidate rank formula carries layer {spec.layer_no} "
                    "outside the selected range"
                )

    def test_no_range_keeps_every_layer(self):
        selection = _select()

        assert {spec.layer_no for spec in selection.local_hooks} == set(range(8))

    def test_range_outside_the_model_selects_nothing(self):
        from dmi.configuration.errors import ConfigValidationError
        from dmi.configuration.schema import LayerSelection

        with pytest.raises(ConfigValidationError, match="matches no hook"):
            _select(layers=LayerSelection(40, 50))

    def test_range_clipped_by_the_model_is_accepted(self):
        from dmi.configuration.schema import LayerSelection

        selection = _select(layers=LayerSelection(6, 50), num_layers=8)

        local_layers = {spec.layer_no for spec in selection.local_hooks}
        assert local_layers == {6, 7}
