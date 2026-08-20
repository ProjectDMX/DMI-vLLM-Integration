# Official vLLM 0.27.1 contract

This integration depends on the following behavior from official vLLM
0.27.1. A vLLM update requires review only when it changes one of these
assumptions or an integrated upstream model definition.

## Model runner and lifecycle

- The V1 GPU model runner is selected; the V2 runner is outside this contract.
- Worker initialization calls `init_device`, `load_model`,
  `compile_or_warm_up_model`, `execute_model`, and `shutdown` in their expected
  lifecycle order.
- `load_model` accepts the keyword-only `load_dummy_weights` argument and
  preserves its meaning.
- The V1 model runner exposes `_prepare_inputs` and
  `_determine_batch_execution_and_padding` with their 0.27.1 signatures.
- One worker process executes at most one model forward at a time.

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
- the sum of scheduled counts is the real packed token-row count; and
- reused input-batch arrays remain valid only when snapshotted at this point.

On a prefix-cache hit, `num_computed_tokens_cpu` identifies the first token
executed by the current forward; cached-prefix activations are absent. Final
logits contain one row per active request in packed request order.

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

## Parallel and model layout

- All ranks in a forward use compatible execution modes and descriptor shapes.
- `get_pp_indices` describes each PP stage's owned layer interval.
- Input token and embedding work belongs to the first PP stage; final
  residual, normalization, and logits work belongs to the last.
- Per-layer tensors belong to their owning PP stage.
- TP head, KV-head, and intermediate partitions determine local tensor shapes.
- Integrated upstream model constructors, forwards, returns, loaders,
  compilation behavior, and parallel semantics remain compatible with the
  corresponding 0.27.1 definitions.

## MoE routing

- `FusedMoERouter.select_experts` returns the expert IDs and weights consumed
  by the following fused-MoE invocation.
- The returned rows retain the token-major input-row order for that router
  invocation.

## Compilation and graph replay

- vLLM/PyTorch compilation retains registered custom-op nodes and their
  declared mutation and alias-ordering dependencies.
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

## Upgrade boundary

Changes to the methods, call order, fields, or semantics above require an
integration review even when their names and signatures remain unchanged.
Changes confined to unrelated vLLM paths or non-integrated model definitions
do not require an integration change.
