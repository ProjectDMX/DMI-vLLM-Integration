"""Focused tests for the general vLLM model-registration plugin."""

from __future__ import annotations

from types import ModuleType

import pytest

from dmi_vllm_integration import plugin


EXPECTED_REGISTRATIONS = {
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
    "DMIMiniMaxM2ForCausalLM": (
        "dmi_vllm_integration.models.minimax_m2:MiniMaxM2PForCausalLM"
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


class FakeModelRegistry:
    models: dict[str, str]
    calls: list[tuple[str, str]]

    @classmethod
    def reset(cls, models: dict[str, str] | None = None) -> None:
        cls.models = {} if models is None else dict(models)
        cls.calls = []

    @classmethod
    def get_supported_archs(cls) -> set[str]:
        return set(cls.models)

    @classmethod
    def register_model(cls, architecture: str, target: str) -> None:
        cls.calls.append((architecture, target))
        cls.models[architecture] = target


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeModelRegistry:
    compat = ModuleType("dmi_vllm_integration.compat")
    compat.calls = 0

    def require_compatible_runtime() -> None:
        compat.calls += 1

    compat.require_compatible_runtime = require_compatible_runtime
    monkeypatch.setitem(__import__("sys").modules, compat.__name__, compat)

    vllm = ModuleType("vllm")
    vllm.ModelRegistry = FakeModelRegistry
    monkeypatch.setitem(__import__("sys").modules, "vllm", vllm)
    FakeModelRegistry.reset({"OfficialArchitecture": "vllm.official:Model"})
    return FakeModelRegistry


def test_register_validates_then_adds_only_lazy_unique_aliases(
    fake_runtime: FakeModelRegistry,
) -> None:
    assert dict(plugin.MODEL_REGISTRATIONS) == EXPECTED_REGISTRATIONS
    assert all(name.startswith("DMI") for name in plugin.MODEL_REGISTRATIONS)

    plugin.register()

    compat = __import__("sys").modules["dmi_vllm_integration.compat"]
    assert compat.calls == 1
    assert dict(fake_runtime.calls) == EXPECTED_REGISTRATIONS
    assert fake_runtime.models["OfficialArchitecture"] == "vllm.official:Model"


def test_register_is_idempotent_and_reasserts_package_owned_aliases(
    fake_runtime: FakeModelRegistry,
) -> None:
    existing = "DMIQwen3ForCausalLM"
    fake_runtime.models[existing] = "third_party.models:UnexpectedModel"

    plugin.register_models()
    plugin.register_models()

    assert all(
        [name for name, _ in fake_runtime.calls].count(architecture) == 2
        for architecture in EXPECTED_REGISTRATIONS
    )
    assert {
        name: fake_runtime.models[name]
        for name in EXPECTED_REGISTRATIONS
    } == EXPECTED_REGISTRATIONS
    assert fake_runtime.models["OfficialArchitecture"] == "vllm.official:Model"
