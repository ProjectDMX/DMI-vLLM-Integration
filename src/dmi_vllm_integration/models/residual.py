"""Where a decoder layer observes ``resid_pre`` (``dmx_resid_pre_read``).

From layer 1 on, the hooked decoder layers see the residual stream entering
the layer as two tensors, ``hidden_states`` and ``residual``, which
``input_layernorm``'s fused add-norm sums and returns as its new
``residual``.  Two observations give the same values (HF's
``hidden_states[layer]``):

``add`` (default)
    Capture ``hidden_states + residual`` before the norm.  The compiled graph
    keeps that extra elementwise add in every layer, even while the hook is
    disabled at runtime by the node toggle.
``returned``
    Capture the residual the norm returns, right after it, so no second add
    exists.  The capture is issued before the attention; the next in-place
    writer of that buffer is ``post_attention_layernorm``, after the
    attention, so the read is stream-ordered before the write.

The mode is read from ``additional_config`` when the model is built.
``additional_config`` is part of vLLM's compilation cache key, so a cached
graph of one mode is never reused for the other.
"""

from __future__ import annotations

from typing import Any, Optional

RESID_PRE_READ_MODES = ("add", "returned")


def resid_pre_from_norm(vllm_config: Optional[Any] = None) -> bool:
    """True when ``resid_pre`` observes the norm's returned residual."""
    if vllm_config is None:
        # Outside a vLLM config context (a layer built directly in a test)
        # the default observation applies.
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
    config = getattr(vllm_config, "additional_config", None)
    mode = config.get("dmx_resid_pre_read", "add") if isinstance(config, dict) else "add"
    if mode not in RESID_PRE_READ_MODES:
        raise ValueError(
            f"dmx_resid_pre_read must be one of {RESID_PRE_READ_MODES}, got {mode!r}")
    return mode == "returned"


__all__ = ["RESID_PRE_READ_MODES", "resid_pre_from_norm"]
