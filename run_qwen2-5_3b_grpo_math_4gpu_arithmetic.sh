#!/usr/bin/env bash
# DATA_SETTING=history_to_aime25 bash run_qwen2-5_3b_grpo_aime_history_4gpu.sh
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_qwen3_4b_grpo_math}"
# export WANDB_PROJECT="${WANDB_PROJECT:-test}"
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
MODEL_PATH="${MODEL_PATH:-/mnt/data/safetyCode/model_hub/Qwen/Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
GSM8K_DIR="${GSM8K_DIR:-$DATA_ROOT/gsm8k}"
MATH_DIR="${MATH_DIR:-$DATA_ROOT/math}"
GSM8K_TRAIN="${GSM8K_TRAIN:-$GSM8K_DIR/train.parquet}"
GSM8K_TEST="${GSM8K_TEST:-$GSM8K_DIR/test.parquet}"
MATH_TRAIN="${MATH_TRAIN:-$MATH_DIR/train.parquet}"
MATH_TEST="${MATH_TEST:-$MATH_DIR/test.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/qwen3_4b_grpo_math_4gpu_arithmetic}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/mnt/data/ray_tmp}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-$RAY_TMP_DIR/object_spill}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ARITHMETIC_GROUP_SIZE="${ARITHMETIC_GROUP_SIZE:-$ROLLOUT_N}"
ARITHMETIC_SEED="${ARITHMETIC_SEED:-9}"

TRAIN_FILES="['$GSM8K_TRAIN', '$MATH_TRAIN']"
VAL_FILES="['$GSM8K_TEST', '$MATH_TEST']"

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
for required_path in "$MODEL_PATH" "$GSM8K_TRAIN" "$GSM8K_TEST" "$MATH_TRAIN" "$MATH_TEST"; do
    if [ ! -e "$required_path" ]; then
        echo "Missing required path: $required_path" >&2
        missing=1
    fi
done

if [ "$missing" -ne 0 ]; then
    cat >&2 <<EOF

Create the math datasets first from the repo root with:
  mkdir -p "$GSM8K_DIR" "$MATH_DIR"
  python3 examples/data_preprocess/gsm8k.py --local_save_dir "$GSM8K_DIR"
  python3 examples/data_preprocess/math_dataset.py --local_save_dir "$MATH_DIR"

Then rerun:
  bash $PROJECT_DIR/run_qwen3_4b_grpo_math_4gpu_arithmetic.sh
EOF
    exit 1
fi

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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.001} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    trainer.experiment_name="${WANDB_RUN_NAME:-qwen3_4b_grpo_math_4gpu_arithmetic}" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMP_DIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-15} \
    "$@"
