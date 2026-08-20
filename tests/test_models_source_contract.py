"""Source-boundary checks for monitored models and validation copies."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


MODEL_DIRECTORY = (
    Path(__file__).parents[1] / "src" / "dmi_vllm_integration" / "models"
)
MODEL_CLASSES = {
    "apertus.py": "ApertusPForCausalLM",
    "ernie45.py": "Ernie4_5PForCausalLM",
    "falcon_h1.py": "FalconH1PForCausalLM",
    "gemma3.py": "Gemma3PForCausalLM",
    "gpt2.py": "GPT2PLMHeadModel",
    "granite.py": "GranitePForCausalLM",
    "jamba.py": "JambaPForCausalLM",
    "lfm2.py": "Lfm2PForCausalLM",
    "llama.py": "LlamaPForCausalLM",
    "minicpm.py": "MiniCPMPForCausalLM",
    "mistral.py": "MistralPForCausalLM",
    "olmo3.py": "Olmo3PForCausalLM",
    "phi3.py": "Phi3PForCausalLM",
    "qwen2.py": "Qwen2PForCausalLM",
    "qwen2_moe.py": "Qwen2MoePForCausalLM",
    "qwen3.py": "Qwen3PForCausalLM",
}
ORACLE_COPY_PROVENANCE = {
    "gpt2_compare.py": (
        "gpt2.py",
        "Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.",
    ),
    "gpt2_ref.py": (
        "gpt2.py",
        "Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.",
    ),
    "llama_compare.py": (
        "llama.py",
        "Copyright 2022 EleutherAI and the HuggingFace Inc. team.",
    ),
    "llama_ref.py": (
        "llama.py",
        "Copyright 2022 EleutherAI and the HuggingFace Inc. team.",
    ),
    "qwen3_ref.py": (
        "qwen3.py",
        "Copyright 2024 The Qwen team.",
    ),
}
ORACLE_DIRECTORY = Path(__file__).parent / "oracles"


@pytest.mark.parametrize(("filename", "model_class"), MODEL_CLASSES.items())
def test_model_port_has_provenance_and_external_import_boundaries(
    filename: str,
    model_class: str,
) -> None:
    source = (MODEL_DIRECTORY / filename).read_text()
    tree = ast.parse(source)

    assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
    upstream_name = filename.removesuffix(".py")
    assert (
        f"Adapted from vllm/model_executor/models/{upstream_name}.py"
        in source
    )
    assert model_class in {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(node.level == 0 for node in imports)
    dmi_imports = {
        node.module
        for node in imports
        if node.module is not None and node.module.startswith("monitoring")
    }
    assert dmi_imports <= {"monitoring.integration_api.v1"}
    if "HookPoint" in source or "HookSpec" in source:
        assert dmi_imports == {"monitoring.integration_api.v1"}

    vllm_imports = {
        node.module
        for node in imports
        if node.module is not None and node.module.startswith("vllm")
    }
    assert any(
        module.startswith("vllm.model_executor.models")
        for module in vllm_imports
    )


@pytest.mark.parametrize(
    ("filename", "upstream_name", "upstream_copyright"),
    tuple(
        (filename, *provenance)
        for filename, provenance in ORACLE_COPY_PROVENANCE.items()
    ),
)
def test_copied_oracle_retains_upstream_license_and_provenance(
    filename: str,
    upstream_name: str,
    upstream_copyright: str,
) -> None:
    source = (ORACLE_DIRECTORY / filename).read_text()

    assert source.startswith("# SPDX-License-Identifier: Apache-2.0")
    assert (
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
        in source
    )
    assert upstream_copyright in source
    assert 'Licensed under the Apache License, Version 2.0 (the "License")' in source
    assert (
        f"Adapted from vllm/model_executor/models/{upstream_name} "
        "in official vLLM 0.27.1."
    ) in source


def test_qwen2_moe_activates_observer_patch_and_observes_one_route() -> None:
    source = (MODEL_DIRECTORY / "qwen2_moe.py").read_text()
    tree = ast.parse(source)

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "apply_fused_moe_router_observer_patch" in calls
    assert "set_routing_observer" in source
    assert source.count(".select_experts(") == 0
