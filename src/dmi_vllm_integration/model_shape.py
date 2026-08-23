"""vLLM-specific normalization for DMI's public model-shape API."""

from __future__ import annotations

from typing import Any

import torch

from dmi_vllm_integration.dmi_api import ModelShapeConfig
from dmi_vllm_integration.dmi_api import (
    make_model_shape_from_hf_config as _make_public_model_shape,
)


def make_model_shape_from_hf_config(
    hf_config: Any,
    dtype: torch.dtype | None = None,
) -> ModelShapeConfig | None:
    """Normalize vLLM decoder/MoE aliases before using DMI API v1.

    Multimodal vLLM wrappers expose the captured decoder as ``text_config``.
    Several supported MoE configs use vLLM-family aliases for expert count
    and top-k.  This compatibility logic belongs to the versioned integration,
    while construction of the framework-neutral shape remains owned by DMI.
    """

    config = getattr(hf_config, "text_config", hf_config)
    shape = _make_public_model_shape(config, dtype=dtype)
    if shape is None:
        return None

    shape.num_experts = int(
        getattr(config, "num_experts", None)
        or getattr(config, "num_local_experts", None)
        or getattr(config, "n_routed_experts", None)
        or 0
    )
    shape.top_k = int(
        getattr(config, "num_experts_per_tok", None)
        or getattr(config, "num_experts_per_token", None)
        or getattr(config, "top_k", None)
        or 0
    )
    return shape


__all__ = ["make_model_shape_from_hf_config"]
