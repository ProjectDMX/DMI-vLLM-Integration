# DMI vLLM integration

`DMI-vLLM-Integration` connects DMI to an unmodified official vLLM
installation. Release `0.27.1` supports exactly vLLM `0.27.1` and requires
DMI integration API v1, first released with DMI `1.1.0`.

```bash
pip install 'DMI>=1.1.0,<2.0'
pip install 'vllm==0.27.1'
pip install 'DMI-vLLM-Integration==0.27.1'
export VLLM_USE_V2_MODEL_RUNNER=0
```

Supported architectures are GPT-2, Llama, Qwen2, Qwen2-MoE, and Qwen3.
An unsupported architecture is rejected before CUDA initialization. When
top-k routing capture is selected for Qwen2-MoE, the loaded routing backend is
validated after model load and before inference; it must expose the modular
routing result consumed by fused MoE.

For offline inference, select DMI's worker through the Python API:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
)
```

The `dmi_models` general plugin registers the integration's model
architectures. Online serving also requires the opt-in finalization endpoint:

```bash
export VLLM_PLUGINS=dmi_models,dmi_stop_monitoring
vllm serve Qwen/Qwen3-0.6B \
    --worker-cls dmi_vllm_integration.worker.DMXGPUWorker
```

After stopping external request intake, call
`POST /v1/dmi/stop_monitoring` before terminating the server. The endpoint
pauses and drains generation, invokes `stop_monitoring` on every worker, and
leaves the engine terminally paused.

The official-vLLM behavior assumed by this release is documented in
[`docs/vllm_contract.md`](https://github.com/ProjectDMX/DMI-vLLM-Integration/blob/v0.27.1/docs/vllm_contract.md).
