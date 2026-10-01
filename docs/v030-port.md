# vLLM 0.30.0 port audit

V1 and V2 eager pass; default compiled V1 and V2 fail stock parity. This is
**not a qualified default-configuration release**. Scope is version compatibility only:
no new models, hook placement changes, weight-tying rewrites or optimizations.

## Identity and planned cells

- Integration base: main `c17b9f2e69b4f330a617896f759313eb331cb6dd` (merged #23).
- Previous upstream: v0.29.0 `98dff2a81d747d1dba01a47f939f48c3526d4206`.
- Target: official v0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- DMI core: 1.2.0 / API v1, `c222f18c4db55f6f2d7136b1d6608c0b540b273a`.
- Separate 0.30 integration worktree and environment; PyTorch remains 2.13.0.
- Planned runtime cells: Qwen3-0.6B BF16, offline generation, TP1/PP1,
  V1/V2 eager and default compilation/CUDA graphs; ClickHouse readback.
- TP2 compiled, other topologies/models, quantization, speculative decoding,
  multimodal input, serving and KV-sharing fast prefill are not qualified here.

## Discovery (2026-10-01)

| Checklist | Boundary | Discovery | Planned action |
| --- | --- | --- | --- |
| C01, R04 | package and official wheel | change-required | Target 0.30.0; do not loosen version policy in this port. |
| G02–G04, G10 | V2 CudaGraphManager.dispatch | change-required | New num_ubatches argument; forward it, preserve ordinary descriptor semantics and existing DBO rejection. |
| G01, G07, S02–S03 | V2 prepare_inputs / InputBatch | compatible for planned cells | Signature, packed request IDs, scheduled and computed counts retained. Removed max_seq_len_np is not consumed by DMI. New fast-prefill layout is outside scope. |
| W01–W07 | worker lifecycle | compatible for planned cells | Wrappers retain inherited lifecycle; V2 kernel warmup moved before graph capture, so validate warmup exclusion in GPU cells. |
| M04, M11 | Qwen3 residual hooks | compatible | Preserve merged-main explicit-add placement and V views exactly. |
| M06–M07 | newly supported PP auxiliary states | outside planned cells | Do not claim speculative PP support from inherited capabilities. |
| M02, M10 | constructors / loaders | compatible at source/import boundary | All 21 registered targets import against the official wheel. GLM inherits the new DeepseekV2 index_group_builder construction; no copied constructor adaptation required. Checkpoint execution outside the cell below is not qualified. |
| N01–N13, L01–L10 | native / storage | validated for bounded cells | Matching native build and actual ClickHouse readback pass; storage success does not override compiled numerical failures. |

## Boundary inventory and model provenance

Regenerate the inventory with `docs/v030-audit-profile.json` and the
`dmi-port-vllm/scripts/audit_vllm_port.py` skill helper. This scan finds **865
occurrences / 555 groups**, with no errors. All 555 group IDs exactly match
the semantic routing in [v029-boundaries.tsv](v029-boundaries.tsv); there are
zero missing or stale IDs. Reuse that routing with the discovery deltas above,
not its historical GPU qualification column. `getattr`-based fast-prefill and
`num_ubatches` contracts require the manual G02/G10 review above even though
the group's identifier did not change. The sole scan warning, no configured
native source paths, is expected for the external integration: the separately
built core is identified above.

Copied model bodies and their original 0.29 provenance headers are unchanged.
They were not relabeled as freshly copied 0.30 implementations. In particular:

- Upstream Qwen3 is unchanged; no residual hook or arithmetic rewrite.
- V1 prepare/dispatch signatures and V2 packed request/count sources remain
  compatible for ordinary decoding. V2's removed `max_seq_len_np` is not read.
- Gemma 4 attention changes and GLM index-group construction are inherited;
  the integration does not duplicate those constructors.
- New PP auxiliary-state transport, quantized activation fusion, new vision
  routing fields and speculative paths are not established by the TP1 text
  cell. Source/import compatibility is not model-wide GPU qualification.
- MoE `select_experts` and the observer call boundary are unchanged. The
  existing version guard is retargeted, not weakened.

## Portable validation

- Python 3.12.8; official `vllm==0.30.0`; torch `2.13.0+cu130`;
  transformers `5.18.0`; flashinfer-python `0.6.18.post1`.
- CPU suite: **691 passed, 4 skipped, 11 deselected**. Three process-global
  MoE patch skips pass individually in fresh processes. The remaining skip
  requires a newer core `dmi.configuration.schema` API than the pinned core.
- All 21 production registration targets import using the real native module,
  not the CPU test stub. This is import evidence, not a model inference claim.
- Native core built against this exact torch/CUDA 13.0 environment; cubin
  checked as `sm_89` for RTX 4090; dynamic libcudart link check passed.
- Integration source distribution and wheel build successfully.

CPU command (native stub is restricted to this portable gate):

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='' DMI_VLLM_TEST_NATIVE_STUB=1 \
  .venv/bin/python -m pytest tests tests_v2 -q -rs \
  -m 'not gpu and not clickhouse and not e2e and not slow'
```

## GPU evidence

RTX 4090 (SM89), Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16, TP1/PP1, 3 ragged prompts,
8 generated tokens each; ClickHouse `25.12.2.54`. Stock and monitored run in
separate processes with a fixed batch barrier and fresh version-specific
compilation caches. No custom-op override or tolerance relaxation.

- **V1 eager PASS:** public text/tokens/finish fields and all 8 raw-logit
  tensors match exactly. 744 persisted rows pass request/layer/token coverage,
  dtype, shape, finiteness, input-token and final-logit argmax checks.
- **V1 compiled/graph FAIL:** both processes generate and the 744 monitored
  rows pass storage checks, but public tokens differ for request 0 at generated
  token 6 (`Italy` versus `France`). First-step raw-logit maximum absolute
  difference is 0.1875. Same resolved configuration: compile mode 3,
  `FULL_AND_PIECEWISE`, `custom_ops=['none']`. No stale cache reuse. Do not
  infer whether this is new to 0.30 or pre-existing without another control.
- Initial **stock V2 eager environmental failure**: another process released
  GPU memory during vLLM profiling (17.47 GiB free became 20.6 GiB), triggering
  the upstream profiling assertion. This is not a DMI compatibility verdict.
  A bounded retry uses the new harness option `--kv-cache-memory-bytes
  1073741824` equally for stock and monitored, bypassing the shared-GPU
  profiling race; the resolved budget is saved in the comparison configuration.
- **V2 eager PASS** with that fixed budget: public output and all 8 raw-logit
  tensors match exactly, 744 stored rows pass the same storage oracle.
- **V2 compiled/graph FAIL** with that fixed budget: public tokens again
  differ at request 0's generated token 6 (stock `France`, monitored `Italy`).
  First-step raw-logit maximum absolute difference is again 0.1875. Both
  configurations match, actual FULL and PIECEWISE CUDA capture is confirmed,
  and 744 monitored rows pass storage checks. This is a transparency failure,
  not a missing-export failure. V2 AOT reload qualification is not attempted
  after this failure; the release gate remains red.
- **Stock stability control PASS:** one V1 compiled rerun loads the stock AOT
  cache (confirmed in logs), reproduces every public output and all 8 raw-logit
  tensors bitwise. The optional KV-budget field added to the harness is
  absent in the original receipt and `None` in the repeat; the runtime budget
  was unset in both. This does not isolate the exact cause of DMI's mismatch.
- **V1 eager residual reference PASS:** all **1,368 residual rows** match
  independent pre-norm old-expression snapshots, including request/layer/range,
  dtype, shape and exact values. **1,416 total stored rows** include token IDs
  and final logits; public output and all 8 raw-logit steps also match stock.
  This diagnostic is explicitly white-box and eager-only, not graph residual
  qualification. Both runs use the same fixed 1 GiB KV budget.

Original evidence directory: `/tmp/v030-smoke.iJQC6i/`. Fixed-budget V2 evidence:
`/tmp/v030-fixed-kv.H3m2W0/`. Failing artifacts are preserved.
The compact [qualification receipt](evidence/v030-qualification.json) records
both successes and failures, resolved configurations, versions and artifact
hashes. The initial shared-memory profiling error is retained in the original
directory's `v2-eager-stock.log`.

This bounded port does not rewrite residual hooks to pursue compiled parity.
The model implementation files and V1 adapter are byte-identical to the merged
base. Fixing/attributing the compiled numerical discrepancy is a separate
remaining blocker, not something waived by the passing portable tests.

Reproduce the main gate with the real matching native backend, cached model,
GPU and ClickHouse available:

```bash
CUDA_VISIBLE_DEVICES=1 DMI_V030_PYTHON="$PWD/.venv/bin/python" \
  DMI_V030_CLEAR_LD_LIBRARY_PATH=1 bash tests/run_v030_smoke.sh
```

The library-path clearing is a host-specific opt-in, not an installation
requirement. The gate intentionally fails at V1 graph today. To reproduce a
later cell independently, run `python -m tests.v030_smoke` for `--mode stock`
and `--mode monitored` with the same `--runner v2`, `--graph` if desired,
`--kv-cache-memory-bytes 1073741824`, and distinct `--output` paths; then use
`--mode compare --stock ... --monitored ...`. Keep separate fresh
`VLLM_CACHE_ROOT` directories for each mode.
