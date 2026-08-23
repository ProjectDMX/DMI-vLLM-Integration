# vLLM behavior assumptions

This integration depends on the following behavior from the target official
vLLM release, including private execution interfaces used to observe the model
work that vLLM actually performs.

## Model runner and lifecycle

- The V1 GPU model runner is selected; the V2 runner is outside this contract.
- Worker initialization calls `init_device`, `load_model`,
  `compile_or_warm_up_model`, `execute_model`, and `shutdown` in their expected
  lifecycle order.
- `load_model` accepts the keyword-only `load_dummy_weights` argument and
  preserves its meaning.
- The V1 model runner exposes `_prepare_inputs` and
  `_determine_batch_execution_and_padding` with the signatures wrapped by the
  integration.
- One worker process executes at most one model forward at a time.
- An `execute_model` update with zero scheduled tokens retires request state
  without invoking the model.
- An EC-transfer non-consumer updates transfer state without preparing or
  executing a local model batch.

## Request layout

For an `execute_model` call that produces a forward, vLLM updates request
state and `input_batch`, prepares inputs, determines final execution and
padding, and then runs the model with that layout.

After `_prepare_inputs` returns:

- `input_batch.req_ids[:input_batch.num_reqs]` is packed model-tensor order;
- scheduled-token counts and `num_computed_tokens_cpu` align with those IDs;
- each request owns one contiguous token-row interval in that order;
- dictionary order in `scheduler_output.num_scheduled_tokens` is not required
  to equal packed order;
- the sum of scheduled counts is the real packed token-row count;
- reused input-batch arrays remain valid only when snapshotted at this point;
  and
- With request-ID randomization enabled, the internal scheduler ID is the
  external request ID followed by `-` and an eight-character hexadecimal
  suffix. With randomization disabled, the ID is unchanged.

On a prefix-cache hit, `num_computed_tokens_cpu` identifies the first token
executed by the current forward; cached-prefix activations are absent. Final
logits on the next-token-only path used for capture contain one row per active
request in packed request order.

## Model input representation

- On the first PP stage, the text-only preprocessing path passes `input_ids`
  and no `inputs_embeds` to the model.
- Enabling prompt embeddings makes preprocessing pass `inputs_embeds` and no
  `input_ids` on the first PP stage.
- An active decoder-only multimodal path prepares `inputs_embeds` and retains
  `input_ids` only when the loaded model declares
  `requires_raw_input_tokens`.
- `MULTIMODAL_REGISTRY.supports_multimodal_inputs` returns false when every
  supported modality has a zero limit, causing that model to use the text-only
  path.

## Dispatch and execution rows

- `_determine_batch_execution_and_padding` returns the `CUDAGraphMode` and
  `BatchDescriptor` used by the following forward.
- `BatchDescriptor.num_tokens` is the execution-row count after padding and
  is at least the real packed token-row count.
- Request order and execution-row count do not change between that boundary
  and model forward without a corresponding new descriptor.
- The caller's eager request is preserved by dispatch.
- Dispatcher decisions account for uniform decode, LoRA, encoder output,
  cascade attention, graph mode, and caller-eager conditions.

PP+SP early dispatch may occur before `_prepare_inputs`; a later post-layout
dispatch still supplies the descriptor used by model forward. Scheduler
maxima bound scheduled tokens and active sequences, CUDA-graph capture sizes
bound graph-padded rows, and SP padding rounds rows to the required TP
multiple.

## Model resolution and construction

- `ModelRegistry` examines the declared architecture list in order, and
  `ModelConfig.architecture` records the architecture vLLM resolved.
- Model loading resolves the current `hf_config.architectures` list, including
  registrations and architecture changes made before `load_model`; its cache
  key includes that architecture tuple.
- `VllmConfig.with_hf_config` supplies a child model configuration, and
  `initialize_model(..., model_class=...)` constructs the explicitly supplied
  model class.
- Upstream model constructors that expose `model_cls`, `_init_model`, or an
  explicit nested-model class honor that construction seam.

## Parallel and model layout

- All ranks in a forward use compatible execution modes and descriptor shapes.
- `get_pp_indices` describes each PP stage's owned layer interval.
- Input token and embedding work belongs to the first PP stage; final
  residual, normalization, and logits work belongs to the last.
- Per-layer tensors belong to their owning PP stage.
- TP head, KV-head, and intermediate partitions determine local tensor shapes.
- Integrated upstream model constructors, forwards, returns, loaders,
  compilation behavior, and parallel semantics remain compatible with the
  corresponding target-release definitions.

## MoE routing

- `FusedMoERouter.select_experts(hidden_states, router_logits,
  topk_indices_dtype=None, *, input_ids=None)` returns
  `(topk_weights, topk_ids)`, the routing result consumed by the following
  fused-MoE invocation.
- The returned rows retain the token-major input-row order for that router
  invocation.
- Without EPLB, returned expert IDs are global logical expert IDs.
- `MoERunner.is_monolithic`, `is_internal_router`,
  `do_naive_dispatch_combine`, `moe_config.pcp_size`, and
  `moe_config.moe_parallel_config.use_all2all_kernels` describe whether routing
  is internal and whether token rows move across ranks before routing.
- Official `FusedMoERouter` does not define `set_routing_observer`; the
  integration can add its process-local observer around the one authoritative
  `select_experts` call.

## Compilation and graph replay

- vLLM/PyTorch compilation retains registered custom-op nodes and their
  declared mutation and alias-ordering dependencies.
- During construction of a `support_torch_compile` model, the compile wrapper
  captures the instance's current bound `forward`; changing the instance class
  afterward does not retarget that compiled callable.
- `CUDAGraphWrapper.unwrap()` returns the underlying runnable used to derive
  the model's hook manifest.
- CUDA-graph replay preserves captured tensor addresses and structural shapes
  while reading current values from tensors updated in place between replays.
- Persistent AOT-cache loading reconstructs those nodes with the same schemas
  and ordering semantics as a cold compile.
- Graph execution uses the request order and row count selected for that
  forward.

## Serving finalization

- Endpoint plugins can retain the server's `EngineClient` and register a
  `/v1` route whose authentication follows the server's API-key configuration.
- `pause_generation(mode="wait", clear_cache=False)` prevents new model work
  and returns after in-flight generation drains.
- `EngineClient.collective_rpc` invokes a named method on every worker and
  waits for completion.
- Worker processes and distributed runtime remain alive until that RPC
  completes and normal server teardown begins.
