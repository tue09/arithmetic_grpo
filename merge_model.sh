
MODEL_PATH="checkpoints/qwen2_5_math_1_5b_grpo_math_paper_4gpu/global_step_1100/actor"

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $MODEL_PATH \
    --target_dir $MODEL_PATH/huggingface