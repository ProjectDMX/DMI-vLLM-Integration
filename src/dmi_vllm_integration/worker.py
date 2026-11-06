"""Public vLLM worker entry point with lazy model-runner selection."""

from __future__ import annotations

from typing import Any

from .adapter import DMXGPUWorker as _DMXV1GPUWorker


# Settings implemented by the V1 worker only (selective capture, node
# toggling, in-process sinks).  The V2 worker would ignore them silently.
V1_ONLY_SETTINGS = (
    "dmx_node_toggle", "dmx_lazy_toggle", "dmx_device_toggle",
    "dmx_enabled_hooks", "dmx_request_gather", "dmx_per_hook_request_gather",
    "dmx_prefill_selection", "dmx_inline_head_slots", "dmx_monitor_capture_mode",
    "dmx_resid_pre_read", "dmx_require_sink", "dmx_batch_submit",
    "dmx_auto_finalize", "dmx_hook_layers",
)


def _refuse_v1_only_features(vllm_config: Any) -> None:
    from .sinks import registered_sink_factory

    config = getattr(vllm_config, "additional_config", None)
    config = config if isinstance(config, dict) else {}
    used = [key for key in V1_ONLY_SETTINGS if config.get(key) not in (None, False, 0, "")]
    if config.get("dmx_resid_pre_read") == "add":
        used.remove("dmx_resid_pre_read")
    if config.get("dmx_monitor_capture_mode") == "dynamic":
        used.remove("dmx_monitor_capture_mode")
    if registered_sink_factory() is not None:
        used.append("register_sink_factory")
    if used:
        raise RuntimeError(
            f"{', '.join(used)} require the V1 model runner; set "
            "VLLM_USE_V2_MODEL_RUNNER=0")


class DMXGPUWorker(_DMXV1GPUWorker):
    """Select the isolated V1 or V2 implementation from vLLM's config."""

    def __new__(cls, *args: Any, **kwargs: Any):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is None:
            # Preserve direct ``__new__`` allocation used by existing V1
            # subclasses and their tests. Normal vLLM construction always
            # supplies ``vllm_config``.
            return super().__new__(cls)

        if vllm_config.use_v2_model_runner:
            if cls is not DMXGPUWorker:
                raise RuntimeError(
                    "V2 model-runner selection does not support subclasses "
                    "of the dynamic DMXGPUWorker entry point; subclass "
                    "DMXV2GPUWorker directly"
                )
            _refuse_v1_only_features(vllm_config)
            from .v2.worker import DMXV2GPUWorker

            return DMXV2GPUWorker(*args, **kwargs)

        return super().__new__(cls)


__all__ = ["DMXGPUWorker"]
