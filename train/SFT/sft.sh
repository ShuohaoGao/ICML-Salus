# 启用调试模式
set -x



# ============================================================
# Read cloud env vars (can be overridden by CLI)
# ============================================================
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-6379}

# Weights & Biases（标准环境变量；训练端需配合 --report_to wandb）
export WANDB_ENTITY=imedllm
export WANDB_PROJECT=SFT

MODEL_LOAD="${MODEL_LOAD:-/model_load}"
model_name="${model_name:-Qwen3-8B}"
dataset_name="${dataset_name:-judge_llm_distill_DPSK_flash_0430}"
CHECKPOINT_SAVE="${CHECKPOINT_SAVE:-output/$model_name}"
NUM_SHARD="${NUM_SHARD:-4}"


ulimit -n 65536


# global_batch_size 随节点数线性扩展（每节点 NUM_SHARD*8）
GLOBAL_BATCH_SIZE=$(($NUM_SHARD * 8 * $WORLD_SIZE))
learning_rate=$(awk "BEGIN {print 2e-5 * $WORLD_SIZE}")
min_lr=$(awk "BEGIN {print 1e-8 * $WORLD_SIZE}")

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NUM_SHARD \
megatron sft \
    --model $MODEL_LOAD/$model_name \
    --dataset data/SFT/${dataset_name}.jsonl \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.02 \
    --tuner_type full \
    --tensor_model_parallel_size 4 \
    --torch_dtype bfloat16 \
    --micro_batch_size 4 \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --recompute_granularity none \
    --num_train_epochs 3 \
    --packing true \
    --finetune true \
    --freeze_llm false \
    --lr $learning_rate \
    --lr_warmup_fraction 0.1 \
    --min_lr 5e-8 \
    --report_to wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_exp_name "${dataset_name}_$(basename "$model_name")_full" \
    --output_dir $CHECKPOINT_SAVE \
    --save_strategy epoch \
    --max_length 8192 \
    --dataloader_num_workers $NUM_SHARD \
    --dataset_num_proc 16 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel false \
    --optimizer_cpu_offload false \
    --use_precision_aware_optimizer true \
    --optimizer_offload_fraction 0