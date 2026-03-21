#!/usr/bin/env bash
# Qwen3-4B-Instruct arithmetic-GRPO run on DAPO-Math-17k with the local 4-GPU
# math setup. This keeps the current arithmetic math hyperparameters and switches
# the dataset/reward wiring to the DAPO parquet format.

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_qwen3_4b_instruct_grpo_dapo_math_17k}"
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
DAPO_DIR="${DAPO_DIR:-$DATA_ROOT/DAPO-Math-17k}"
DAPO_TRAIN="${DAPO_TRAIN:-$DAPO_DIR/data/dapo-math-17k.parquet}"
AIME_DIR="${AIME_DIR:-$DATA_ROOT/aime}"
AIME23_FILE="${AIME23_FILE:-$AIME_DIR/aime-history-2023.parquet}"
AIME24_FILE="${AIME24_FILE:-$AIME_DIR/aime-2024.parquet}"
AIME25_FILE="${AIME25_FILE:-$AIME_DIR/aime-2025.parquet}"
RUN_TAG="${RUN_TAG:-qwen3_4b_instruct_grpo_dapo_math_17k_4gpu_arithmetic_clip_ratio_high_0.24_QAE_adv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/$RUN_TAG}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/mnt/data/ray_tmp}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-$RAY_TMP_DIR/object_spill}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ARITHMETIC_GROUP_SIZE="${ARITHMETIC_GROUP_SIZE:-$ROLLOUT_N}"
ARITHMETIC_SEED="${ARITHMETIC_SEED:-9}"
ARITHMETIC_PROBE_COUNT="${ARITHMETIC_PROBE_COUNT:-2}"
ARITHMETIC_PASS_REWARD_THRESHOLD="${ARITHMETIC_PASS_REWARD_THRESHOLD:-0.0}"

TRAIN_FILES="['$DAPO_TRAIN']"
VAL_FILES="['$AIME23_FILE', '$AIME24_FILE', '$AIME25_FILE']"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RAY_TMP_DIR" "$RAY_SPILL_DIR"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$FLASHINFER_WORKSPACE_BASE"

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

missing=0
missing_aime23=0
missing_aime24=0
missing_aime25=0
for required_path in "$MODEL_PATH" "$DAPO_TRAIN" "$AIME23_FILE" "$AIME24_FILE" "$AIME25_FILE"; do
    if [ ! -e "$required_path" ]; then
        echo "Missing required path: $required_path" >&2
        missing=1
        case "$required_path" in
            "$AIME23_FILE")
                missing_aime23=1
                ;;
            "$AIME24_FILE")
                missing_aime24=1
                ;;
            "$AIME25_FILE")
                missing_aime25=1
                ;;
        esac
    fi
done

if [ "$missing" -ne 0 ]; then
    cat >&2 <<EOF

Required data is missing.

Expected training parquet:
  $DAPO_TRAIN

Expected validation parquets:
  $AIME23_FILE
  $AIME24_FILE
  $AIME25_FILE

To download DAPO-Math-17k into the default location:
  mkdir -p "$DAPO_DIR/data"
  curl -L --fail --output "$DAPO_TRAIN" \\
    https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet

If AIME-2023 is missing, recreate the historical AIME datasets from the repo root with:
  mkdir -p "$AIME_DIR"
  python3 examples/data_preprocess/aime_history_dataset.py --local_save_dir "$AIME_DIR"

If AIME-2024 / AIME-2025 are missing, recreate them from the repo root with:
  mkdir -p "$AIME_DIR"
  python3 examples/data_preprocess/aime_dataset.py --datasets aime24 aime25 --local_save_dir "$AIME_DIR"

Then rerun:
  bash $PROJECT_DIR/run_qwen3_4b_instruct_grpo_dapo_math_17k_4gpu_arithmetic.sh
EOF
    exit 1
fi

echo "TRAIN_FILES=$TRAIN_FILES"
echo "VAL_FILES=$VAL_FILES"
echo "MODEL_PATH=$MODEL_PATH"
echo "ROLLOUT_N=$ROLLOUT_N"
echo "ARITHMETIC_GROUP_SIZE=$ARITHMETIC_GROUP_SIZE"
echo "ARITHMETIC_SEED=$ARITHMETIC_SEED"
echo "ARITHMETIC_PROBE_COUNT=$ARITHMETIC_PROBE_COUNT"
echo "ARITHMETIC_PASS_REWARD_THRESHOLD=$ARITHMETIC_PASS_REWARD_THRESHOLD"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-512} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-2048} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    ++actor_rollout_ref.model.override_config.attn_implementation="$TRAIN_ATTN_IMPLEMENTATION" \
    actor_rollout_ref.actor.clip_ratio_high=0.24 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-16} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-False} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-16} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-2} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.5} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-1.0} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-2560} \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.enable=True \
    ++actor_rollout_ref.rollout.arithmetic_sampling.group_size="$ARITHMETIC_GROUP_SIZE" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.seed="$ARITHMETIC_SEED" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.probe_count="$ARITHMETIC_PROBE_COUNT" \
    ++actor_rollout_ref.rollout.arithmetic_sampling.pass_reward_threshold="$ARITHMETIC_PASS_REWARD_THRESHOLD" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.6} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE:-True} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-16} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++critic.model.override_config.attn_implementation="$TRAIN_ATTN_IMPLEMENTATION" \
    reward.reward_manager.name=dapo \
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
    trainer.save_freq=${SAVE_FREQ:-500} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-20} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-2000} \
    "$@"
