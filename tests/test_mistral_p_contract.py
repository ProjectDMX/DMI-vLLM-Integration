"""CPU contracts for the Mistral DMI variant."""

from types import SimpleNamespace

import pytest

from dmi_vllm_integration.architectures import ARCHITECTURE_REMAP, LLAMA_COMPAT_ARCHITECTURES
from dmi_vllm_integration.models.llama import LlamaPForCausalLM
from vllm.model_executor.models.mistral import MistralForCausalLM
from tests.oracles.mistral_compare import (
    MistralCompareForCausalLM,
)
from dmi_vllm_integration.models.mistral import (
    MistralPForCausalLM,
    _reject_unsupported_mistral_branches,
)


pytestmark = pytest.mark.vllm


def _vllm_config(**config_values):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**config_values))
    )


def test_mistral_has_a_distinct_dmi_remap() -> None:
    assert "MistralForCausalLM" not in LLAMA_COMPAT_ARCHITECTURES
    assert ARCHITECTURE_REMAP["MistralForCausalLM"] == "DMIMistralForCausalLM"


def test_mistral_variant_preserves_upstream_class_contracts() -> None:
    assert issubclass(MistralPForCausalLM, LlamaPForCausalLM)
    assert MistralPForCausalLM.embedding_modules == {}
    assert (
        MistralPForCausalLM.mistral_mapping
        == MistralForCausalLM.mistral_mapping
    )
    assert (
        MistralPForCausalLM.packed_modules_mapping
        == MistralForCausalLM.packed_modules_mapping
    )
    assert (
        MistralPForCausalLM.maybe_remap_mistral
        is MistralForCausalLM.maybe_remap_mistral
    )
    assert (
        MistralPForCausalLM.get_hook_specs
        is LlamaPForCausalLM.get_hook_specs
    )
    assert MistralCompareForCausalLM.embedding_modules == {}


@pytest.mark.parametrize(
    "config_values, branch",
    [
        ({"llama_4_scaling": {"beta": 0.1}}, "llama_4_scaling"),
        ({"ada_rms_norm_t_cond": True}, "ada_rms_norm_t_cond"),
    ],
)
def test_mistral_fails_closed_on_unaudited_math_branches(
    config_values, branch
) -> None:
    with pytest.raises(NotImplementedError, match=branch):
        _reject_unsupported_mistral_branches(_vllm_config(**config_values))


def test_mistral_accepts_the_v02_standard_config_contract() -> None:
    _reject_unsupported_mistral_branches(
        _vllm_config(
            llama_4_scaling=None,
            ada_rms_norm_t_cond=False,
        )
    )
