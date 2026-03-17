#!/usr/bin/env bash
# DATA_SETTING=history_to_aime25 bash run_qwen3_4b_grpo_aime_history_4gpu_arithmetic.sh

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/mnt/data/safetyCode/flashinfer}"
export TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-sdpa}"
unset TRANSFORMERS_CACHE || true

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$PYTHON_BIN"
elif [ -x "/home/clouduser/miniconda/envs/verl-qwen3/bin/python" ]; then
    PYTHON_BIN="/home/clouduser/miniconda/envs/verl-qwen3/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi
export VERL_RAY_PY_EXECUTABLE="$PYTHON_BIN"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$PROJECT_DIR/vendor${PYTHONPATH:+:$PYTHONPATH}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/safetyCode/model_hub/Qwen/Qwen3-4B-Instruct-2507}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
AIME_DIR="${AIME_DIR:-$DATA_ROOT/aime}"
AIME_HISTORY_1983_2022="${AIME_HISTORY_1983_2022:-$AIME_DIR/aime-history-1983-2022.parquet}"
AIME_HISTORY_1983_2023="${AIME_HISTORY_1983_2023:-$AIME_DIR/aime-history-1983-2023.parquet}"
AIME_HISTORY_1983_2024="${AIME_HISTORY_1983_2024:-$AIME_DIR/aime-history-1983-2024.parquet}"
AIME_HISTORY_2023="${AIME_HISTORY_2023:-$AIME_DIR/aime-history-2023.parquet}"
AIME_HISTORY_2024="${AIME_HISTORY_2024:-$AIME_DIR/aime-history-2024.parquet}"
AIME24_FILE="${AIME24_FILE:-$AIME_DIR/aime-2024.parquet}"
AIME25_FILE="${AIME25_FILE:-$AIME_DIR/aime-2025.parquet}"
DATA_SETTING="${DATA_SETTING:-history_to_aime24}"
TRAIN_DATA_PRESET="${TRAIN_DATA_PRESET:-}"
VAL_DATA_PRESET="${VAL_DATA_PRESET:-}"
TRAIN_FILES_OVERRIDE="${TRAIN_FILES:-}"
VAL_FILES_OVERRIDE="${VAL_FILES:-}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/mnt/data/ray_tmp}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-$RAY_TMP_DIR/object_spill}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ARITHMETIC_GROUP_SIZE="${ARITHMETIC_GROUP_SIZE:-$ROLLOUT_N}"
ARITHMETIC_SEED="${ARITHMETIC_SEED:-9}"

DEFAULT_TRAIN_DATA_PRESET="aime_history_1983_2023"
DEFAULT_VAL_DATA_PRESET="aime24"

case "$DATA_SETTING" in
    history_to_aime24)
        ;;
    history_to_aime25)
        DEFAULT_TRAIN_DATA_PRESET="aime_history_1983_2022"
        DEFAULT_VAL_DATA_PRESET="aime23_aime24_aime25"
        ;;
    history_to_aime24_aime25)
        DEFAULT_VAL_DATA_PRESET="aime24_aime25"
        ;;
    *)
        echo "Unsupported DATA_SETTING: $DATA_SETTING" >&2
        echo "Supported values: history_to_aime24, history_to_aime25, history_to_aime24_aime25" >&2
        exit 1
        ;;
esac

TRAIN_DATA_PRESET="${TRAIN_DATA_PRESET:-$DEFAULT_TRAIN_DATA_PRESET}"
VAL_DATA_PRESET="${VAL_DATA_PRESET:-$DEFAULT_VAL_DATA_PRESET}"

format_hydra_list() {
    local result="["
    local separator=""
    local item

    for item in "$@"; do
        result="${result}${separator}'${item}'"
        separator=", "
    done

    result="${result}]"
    printf '%s' "$result"
}

resolve_data_preset() {
    local preset="$1"

    case "$preset" in
        aime_history_1983_2022)
            printf '%s\n' "$AIME_HISTORY_1983_2022"
            ;;
        aime_history_1983_2023)
            printf '%s\n' "$AIME_HISTORY_1983_2023"
            ;;
        aime_history_1983_2024)
            printf '%s\n' "$AIME_HISTORY_1983_2024"
            ;;
        aime_history_2023)
            printf '%s\n' "$AIME_HISTORY_2023"
            ;;
        aime_history_2024)
            printf '%s\n' "$AIME_HISTORY_2024"
            ;;
        aime24)
            printf '%s\n' "$AIME24_FILE"
            ;;
        aime25)
            printf '%s\n' "$AIME25_FILE"
            ;;
        aime24_aime25 | aime25_aime24)
            printf '%s\n' "$AIME24_FILE" "$AIME25_FILE"
            ;;
        aime23_aime24_aime25 | aime25_aime24_aime23)
            printf '%s\n' "$AIME_HISTORY_2023" "$AIME24_FILE" "$AIME25_FILE"
            ;;
        *)
            echo "Unsupported data preset: $preset" >&2
            return 1
            ;;
    esac
}

if [ -z "$TRAIN_FILES_OVERRIDE" ]; then
    TRAIN_PATHS_RAW="$(resolve_data_preset "$TRAIN_DATA_PRESET")"
    mapfile -t TRAIN_PATHS <<<"$TRAIN_PATHS_RAW"
    TRAIN_FILES="$(format_hydra_list "${TRAIN_PATHS[@]}")"
else
    TRAIN_FILES="$TRAIN_FILES_OVERRIDE"
fi

if [ -z "$VAL_FILES_OVERRIDE" ]; then
    VAL_PATHS_RAW="$(resolve_data_preset "$VAL_DATA_PRESET")"
    mapfile -t VAL_PATHS <<<"$VAL_PATHS_RAW"
    VAL_FILES="$(format_hydra_list "${VAL_PATHS[@]}")"
else
    VAL_FILES="$VAL_FILES_OVERRIDE"
fi

print_dataset_sizes() {
    local label="$1"
    local files_literal="$2"

    DATASET_LABEL="$label" DATASET_FILES_LITERAL="$files_literal" "$PYTHON_BIN" - <<'PY'
import ast
import os
import sys

try:
    import pyarrow.parquet as pq
except Exception as exc:
    print(f"{os.environ['DATASET_LABEL']} dataset size summary skipped: {exc}", file=sys.stderr)
    raise SystemExit(0)

label = os.environ["DATASET_LABEL"]
files_literal = os.environ["DATASET_FILES_LITERAL"]
try:
    paths = ast.literal_eval(files_literal)
except Exception:
    paths = files_literal
if isinstance(paths, str):
    paths = [paths]
total = 0

print(f"{label} dataset sizes:")
for path in paths:
    rows = pq.ParquetFile(path).metadata.num_rows
    total += rows
    print(f"  {path}: {rows}")
print(f"{label} dataset total: {total}")
PY
}

RUN_TAG="${RUN_TAG:-qwen3_4b_grpo_${TRAIN_DATA_PRESET}_to_${VAL_DATA_PRESET}_arithmetic}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_qwen3_4b_grpo_aime_history}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/$RUN_TAG}"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RAY_TMP_DIR" "$RAY_SPILL_DIR"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$FLASHINFER_WORKSPACE_BASE" "$AIME_DIR"

"$PYTHON_BIN" - <<'PY'
from packaging.version import Version
import sys
import numpy
import transformers

errors = []
if Version(transformers.__version__) < Version("4.55.2") or Version(transformers.__version__) >= Version("5.0.0"):
    errors.append(
        f"transformers=={transformers.__version__} is incompatible with this verl+vLLM setup. "
        "Install a 4.x release, for example: pip install --upgrade --force-reinstall "
        "'transformers[hf_xet]>=4.55.2,<5.0.0'"
    )
if Version(numpy.__version__) >= Version("2.0.0"):
    print(
        f"numpy=={numpy.__version__} is incompatible with this verl checkout. "
        "Continuing anyway because this environment previously ran this training setup. "
        "If you later hit NumPy-related runtime errors, install: "
        "pip install --upgrade --force-reinstall 'numpy<2.0.0'",
        file=sys.stderr,
    )

if errors:
    raise SystemExit("\n".join(errors))
PY

required_paths=("$MODEL_PATH")

if [ -z "$TRAIN_FILES_OVERRIDE" ]; then
    required_paths+=("${TRAIN_PATHS[@]}")
fi

if [ -z "$VAL_FILES_OVERRIDE" ]; then
    required_paths+=("${VAL_PATHS[@]}")
fi

missing=0
missing_history=0
missing_eval=0

for required_path in "${required_paths[@]}"; do
    if [ ! -e "$required_path" ]; then
        echo "Missing required path: $required_path" >&2
        missing=1
        case "$required_path" in
            "$AIME_HISTORY_1983_2022" | "$AIME_HISTORY_1983_2023" | "$AIME_HISTORY_1983_2024" | "$AIME_HISTORY_2023" | "$AIME_HISTORY_2024")
                missing_history=1
                ;;
            "$AIME24_FILE" | "$AIME25_FILE")
                missing_eval=1
                ;;
        esac
    fi
done

if [ "$missing" -ne 0 ]; then
    echo >&2
    echo "Current dataset selection:" >&2
    echo "  DATA_SETTING=$DATA_SETTING" >&2
    echo "  TRAIN_DATA_PRESET=$TRAIN_DATA_PRESET" >&2
    echo "  VAL_DATA_PRESET=$VAL_DATA_PRESET" >&2
    echo "  TRAIN_FILES=$TRAIN_FILES" >&2
    echo "  VAL_FILES=$VAL_FILES" >&2
    echo >&2

    if [ "$missing_history" -eq 1 ]; then
        cat >&2 <<EOF
Create the historical AIME datasets from the repo root with:
  mkdir -p "$AIME_DIR"
  python3 examples/data_preprocess/aime_history_dataset.py --local_save_dir "$AIME_DIR"

EOF
    fi

    if [ "$missing_eval" -eq 1 ]; then
        cat >&2 <<EOF
Create the AIME24 / AIME25 evaluation datasets from the repo root with:
  mkdir -p "$AIME_DIR"
  python3 examples/data_preprocess/aime_dataset.py --datasets aime24 aime25 --local_save_dir "$AIME_DIR"

EOF
    fi

    cat >&2 <<EOF
Then rerun, for example:
  DATA_SETTING=history_to_aime24 bash $PROJECT_DIR/run_qwen3_4b_grpo_aime_history_4gpu_arithmetic.sh
  DATA_SETTING=history_to_aime25 bash $PROJECT_DIR/run_qwen3_4b_grpo_aime_history_4gpu_arithmetic.sh
EOF
    exit 1
fi

echo "DATA_SETTING=$DATA_SETTING"
echo "TRAIN_DATA_PRESET=$TRAIN_DATA_PRESET"
echo "VAL_DATA_PRESET=$VAL_DATA_PRESET"
echo "TRAIN_FILES=$TRAIN_FILES"
echo "VAL_FILES=$VAL_FILES"
echo "ROLLOUT_N=$ROLLOUT_N"
echo "ARITHMETIC_GROUP_SIZE=$ARITHMETIC_GROUP_SIZE"
echo "ARITHMETIC_SEED=$ARITHMETIC_SEED"
print_dataset_sizes "Train" "$TRAIN_FILES"
print_dataset_sizes "Val" "$VAL_FILES"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-256} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-512} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-2048} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    ++actor_rollout_ref.model.override_config.attn_implementation="$TRAIN_ATTN_IMPLEMENTATION" \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-2} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.5} \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.enable=True \
    ++actor_rollout_ref.rollout.arithmetic_sampling.group_size="$ARITHMETIC_GROUP_SIZE" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.seed="$ARITHMETIC_SEED" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-8} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++critic.model.override_config.attn_implementation="$TRAIN_ATTN_IMPLEMENTATION" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="${WANDB_RUN_NAME:-$RUN_TAG}" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMP_DIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-50} \
    "$@"
