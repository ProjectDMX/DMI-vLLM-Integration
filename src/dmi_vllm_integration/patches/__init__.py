"""Narrow, version-gated compatibility patches for official vLLM."""

from dmi_vllm_integration.patches.fused_moe_router import (
    apply_fused_moe_router_observer_patch,
)

__all__ = ["apply_fused_moe_router_observer_patch"]
