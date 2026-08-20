#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

HOOKS="${E2E_HOOK_SELECTION:-vllm-full}"
TP="${E2E_TP_SIZE:-2}"
MAX_MODEL_LEN="${E2E_MAX_MODEL_LEN:-128}"
MAX_BATCHED="${E2E_MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEM_UTIL="${E2E_GPU_MEM_UTIL:-0.8}"
DTYPE="${E2E_DTYPE:-bfloat16}"
ENFORCE_EAGER="${E2E_ENFORCE_EAGER:-1}"
MOE_BACKEND="${E2E_MOE_BACKEND:-triton}"
DB_HOST="${DMX_DB_HOST:-localhost}"
DB_PORT="${DMX_DB_PORT:-9000}"
RING_PAYLOAD_MB="${E2E_RING_PAYLOAD_MB:-4096}"
RING_PINNED_MB="${E2E_RING_PINNED_MB:-4096}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_BASE="${E2E_ARTIFACT_BASE:-$ROOT_DIR/artifacts}"
RUN_DIR="${E2E_ARTIFACT_DIR:-$ARTIFACT_BASE/vllm_qwen2_moe_pipeline_$TIMESTAMP}"

mkdir -p "$RUN_DIR/compare"

export VLLM_USE_V2_MODEL_RUNNER=0
export E2E_MODEL=qwen2_moe
export E2E_TP_SIZE="$TP"
export E2E_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export E2E_MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED"
export E2E_GPU_MEM_UTIL="$GPU_MEM_UTIL"
export E2E_DTYPE="$DTYPE"
export E2E_ENFORCE_EAGER="$ENFORCE_EAGER"
export E2E_MOE_BACKEND="$MOE_BACKEND"
export DMX_HOOK_SELECTION="$HOOKS"
export DMX_DB_HOST="$DB_HOST"
export DMX_DB_PORT="$DB_PORT"
export E2E_RING_PAYLOAD_MB="$RING_PAYLOAD_MB"
export E2E_RING_PINNED_MB="$RING_PINNED_MB"
export COMPARE_OUTPUT_DIR="$RUN_DIR/compare"
export COMPARE_RESULT_FILE="$RUN_DIR/result.json"

{
  echo "============================================"
  echo "vLLM MoE pipeline"
  echo "model=qwen2_moe"
  echo "hooks=$HOOKS"
  echo "tp=$TP"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_batched_tokens=$MAX_BATCHED"
  echo "gpu_memory_utilization=$GPU_MEM_UTIL"
  echo "moe_backend=$MOE_BACKEND"
  echo "dtype=$DTYPE"
  echo "run_dir=$RUN_DIR"
  echo "============================================"
} | tee "$RUN_DIR/summary.log"

python -m tests.vllm_compare_runner 2>&1 | tee "$RUN_DIR/compare.log"

echo "done: $RUN_DIR"
