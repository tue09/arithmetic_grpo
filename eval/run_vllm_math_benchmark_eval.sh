#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_PATH=${1:?Usage: bash verl/eval/run_vllm_math_benchmark_eval.sh /path/to/model [extra args...]}
shift

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-verl-qwen3}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

python "${SCRIPT_DIR}/vllm_math_benchmark_eval.py" \
  --model "${MODEL_PATH}" \
  --benchmarks AIME24 AIME25 AMC23 MATH500 Minerva Olympiad \
  --k 1 2 4 8 16 32 \
  --temperature "${TEMPERATURE:-0.6}" \
  --top-p "${TOP_P:-0.95}" \
  --top-k "${TOP_K:-0}" \
  --max-tokens "${MAX_TOKENS:-2048}" \
  --sample-batch-size "${SAMPLE_BATCH_SIZE:-4}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
  --disable-custom-all-reduce \
  --dtype "${DTYPE:-auto}" \
  --seed "${SEED:-0}" \
  --output-dir "${OUTPUT_DIR:-${SCRIPT_DIR}/data}" \
  "$@"
