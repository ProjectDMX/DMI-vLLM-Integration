"""Public vLLM worker entry point with lazy model-runner selection."""

from __future__ import annotations

from typing import Any

from .adapter import DMXGPUWorker as _DMXV1GPUWorker


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
            from .v2.worker import DMXV2GPUWorker

            return DMXV2GPUWorker(*args, **kwargs)

        return super().__new__(cls)


__all__ = ["DMXGPUWorker"]
