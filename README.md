# DMI vLLM integration

`DMI-vLLM-Integration` connects DMI to an unmodified official vLLM
installation. Release `0.27.1` supports exactly vLLM `0.27.1` and requires
DMI integration API v1, first released with DMI `1.1.0`.

Choose a [DMI release tag](https://github.com/ProjectDMX/DMI/tags) in the
range `>=v1.1.0,<v2.0.0`, then follow the `docs/install.md` shipped in that
checkout, including its native-backend build. Next, choose the latest immutable
[integration tag](https://github.com/ProjectDMX/DMI-vLLM-Integration/tags)
that targets vLLM `0.27.1`. From that integration checkout, run:

```bash
python -m pip install 'vllm==0.27.1'
python -m pip install .
export VLLM_USE_V2_MODEL_RUNNER=0
```

The integration package is distributed from this source repository and its
immutable tags; it is not published to PyPI or another package registry.

## Model support

The following model families are available in this source tree. Experimental
entries have not completed real-checkpoint GPU qualification:

- Apertus
- DeepSeek V4 Flash (experimental)
- ERNIE 4.5 (dense)
- Gemma 3 (text)
- Gemma 4 E2B (text)
- GLM-5.2 (experimental)
- GPT-2
- GPT-OSS
- Granite 4.1
- Kimi K3 (experimental)
- Llama
- Llama 4 (experimental)
- MiniCPM 4.1 (dense)
- MiniMax-M2.7 (experimental)
- Mistral
- OLMo 3
- Phi-3.5
- Qwen2
- Qwen2-MoE
- Qwen3
- Qwen3-MoE
- Qwen3.6 (text)

Gemma 3, Gemma 4 E2B, and Qwen3.6 currently claim text inference only.
An unsupported architecture is rejected before CUDA initialization. When
top-k routing capture is selected for an MoE model, the loaded routing
backend is validated after model load and before inference; it must expose the
modular routing result consumed by fused MoE.

For offline inference, select DMI's worker through the Python API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
)

try:
    outputs = llm.generate(
        ["The answer is"],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    print(outputs[0].outputs[0].text)
finally:
    llm.collective_rpc("stop_monitoring")
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

```bash
curl --fail-with-body -X POST \
  'http://127.0.0.1:8000/v1/dmi/stop_monitoring?timeout=30'
```

If the server uses an API key, add
`-H "Authorization: Bearer $VLLM_API_KEY"` to that request.

The official-vLLM behavior assumed by this release is documented in
[`docs/vllm_contract.md`](docs/vllm_contract.md).
