# Selective capture, node toggling and in-process consumers

These features run on the V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`);
the dynamic worker entry point refuses them under V2. They need a DMI build
with CUDA-graph node toggling, per-HookPoint row tables and in-process ring
consumers (`dmi.node_toggle`, `dmi.hook_rows`, `MonitoringEngine(ring_consumer=...)`).
With none of the settings below, capture is unchanged.

## In-process consumers

vLLM builds the worker and passes it only serializable configuration, so a
live consumer is registered as a factory before the engine is constructed:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # worker in this process

from dmi.transport import native
from dmi_vllm_integration.sinks import register_sink_factory

class Consumer:
    def __init__(self):
        self.ring_sink = native.BatchQueue(256 << 20)  # GIL-free native queue
        self.dmx_batch_submit = True                  # whole-hook handoff
    def finalize_request(self, request_id, valid=True):
        ...                                           # all rows delivered

consumer = Consumer()
register_sink_factory(lambda: consumer)
llm = LLM(model, worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
          additional_config={"dmx_hook_selection": "resid_pre"})
```

With no `dmx_db_host`, the worker calls the factory once. The object's
`ring_sink` (a `SliceQueue` or `BatchQueue`) is the ring's consumer; without
one the object itself is called once per request slice. The caller owns the
consumer: the worker never starts or stops it. `dmx_require_sink=True` makes
a missing factory an error instead of a sink-less ring.

### Request completion

When the consumer has `finalize_request`, each finished request is finalized
after a delivery barrier (`drain_to_sink_and_wait`) confirms its last rows
reached the consumer. A barrier that times out is retried on later steps
(`dmx_finalize_drain_timeout_ms`, default 200; `dmx_finalize_max_steps`,
default 8), then the request is finalized `valid=False`; a consumer failure
does so at once. vLLM reports a finish only in a later scheduler step, so for
requests finishing in the last step call
`llm.collective_rpc("dmx_finalize_requests", args=(request_ids,))`.
`stop_monitoring` delivers the tail (bounded by
`dmx_finalize_sweep_timeout_ms`, default 5000; on timeout it raises and can be
retried), finalizes pending requests, and finalizes requests never reported
finished `valid=False`. `dmx_auto_finalize=False` turns this off.

## Layer sets

`dmx_hook_layers="18,20,22,31"` installs per-layer hooks only at those layers
(global hooks such as `final_ln` always stay). Unlike a node toggle, the other
layers' producers never enter a graph. The byte formulas used for the eager
preflight count the same layers.

## Node toggling

| setting | meaning |
|---|---|
| `dmx_node_toggle` | record producer nodes in the warmup captures |
| `dmx_lazy_toggle` | apply node state per graph at its next replay |
| `dmx_device_toggle` | device-side toggling of row-table producers |
| `dmx_enabled_hooks` | static set: `"hook_type:layer,..."`, `"none"`, or unset (all on) |

Only full-model graphs (`CUDAGraphMode.FULL`) are registered; piecewise
segment captures are excluded. Each step is classified from vLLM's real
dispatch: a step replaying a full graph publishes metadata and reserves ring
space for the enabled hooks only; an eager or piecewise step for every
installed hook (unless the row-table gate below is on). An enabled set with no
full graph captured is refused.

## Request and HookPoint selection

| setting | meaning |
|---|---|
| `dmx_request_gather` | one shared row table: masked requests' rows are not copied |
| `dmx_per_hook_request_gather` | one row table per HookPoint |
| `dmx_prefill_selection` | the tables' disabled marker also gates eager and piecewise producers |
| `dmx_table_span_upload` | copy only the rewritten span of the tables (default on) |

RPCs (via `llm.collective_rpc`):

* `dmx_set_request_mask(allowlist, capture_new=True)`: capture only these
  requests (and, with `capture_new`, requests scheduled later); `None`
  captures all. A request masked while in flight is finalized `valid=False`.
* `dmx_set_hook_request_selection(rows, default=None, token_windows=None)`:
  `rows` maps request IDs to `(hook_type, layer)` pairs; `token_windows`
  restricts pairs to absolute half-open positions, e.g.
  `{"req": {(0, 18): [(61, 64)]}}` selects the last three prompt rows of a
  64-token prompt.
* `dmx_request_mask_state()`: configuration and cumulative counters.

### Capture planners

A consumer may instead expose `capture_planner`, an object planning demand
every step (`capture_plan(req_ids, token_ranges)` returning `rows`/`windows`
and an optional `capture_lease`; `admit_capture_inventory(inventory)` once at
attachment). It requires per-HookPoint gather, prefill selection and lazy node
toggling: the planned HookPoints become the enabled set, and the node state
follows at each graph's next replay. `dmx_monitor_capture_mode="static_full"`
keeps full capture and only attaches the planner's leases. See
`capture_selection.py` for the protocol.

## In-graph projection heads

`dmx_inline_head_slots=K` (1..16, requires per-HookPoint gather) adds a fused
head node after every `resid_pre` and `final_ln` HookPoint: `K` fp32 affine
scores per selected row, computed inside the graph, under the head's own
HookPoint (`resid_pre_head`, `final_ln_head`). Select it like any HookPoint.
`dmx_inline_head_weight` is `fp32` (default) or `bf16`.
`dmx_inline_register(entries)` writes or retires slots between steps without
recapture; a linear head over several layers registers `first` ... `last`
slots sharing one accumulator column.

## Residual read

`dmx_resid_pre_read="returned"` makes the Llama and Qwen3 layers observe
`resid_pre` as the residual their fused input norm returns, instead of
recomputing `hidden_states + residual` (`"add"`, default). The values are the
same; the compiled graph loses one elementwise add per layer. The setting is
part of vLLM's compilation cache key.

## Telemetry

`dmx_sink_stats()` reports consumer failures, finalize-barrier waits, the
consumer's `drop_stats()`, and DMI's host telemetry (`dmi.sync_stats`);
`dmx_ring_stats()` the payload ring's capacity and occupancy.
