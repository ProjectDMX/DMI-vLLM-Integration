"""DMI-monitored model implementations for the supported vLLM release.

Modules stay lazy: vLLM's model registry imports only the architecture selected
for a worker, avoiding eager CUDA and model-module initialization.
"""

__all__ = ["gpt2", "llama", "qwen2", "qwen2_moe", "qwen3"]
