# Visualization example

With ClickHouse running, capture the `demo_vllm` slot for DMI's visualization
notebook:

```bash
python examples/visualization/run_offload.py
```

The script uses Qwen3-0.6B and removes only its own prior `demo_vllm` rows.
