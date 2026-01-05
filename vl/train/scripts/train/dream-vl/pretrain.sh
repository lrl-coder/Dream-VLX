export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

LLM_VERSION="Dream-org/Dream-v0-Instruct-7B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="hf:Emova-ollm/qwen2vit600m"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
############### Pretrain ################
# export HF_HOME=xxx

export WANDB_API_KEY=
# wandb login --relogin $WANDB_API_KEY
export WANDB_PROJECT=Dream-VL

PROMPT_VERSION=plain

module load cuda12.2/toolkit/12.2.2 cuda12.2/nsight/12.2.2 cuda12.2/profiler/12.2.2
NUM_GPUS=8
NNODES=$SLURM_NNODES
WORLD_SIZE=$((NNODES * NUM_GPUS))
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

BASE_RUN_NAME="dream-vl_pretrain_qwen2vit"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /path_to/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /path_to/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio native_anyres \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /path_to/LLaVA-Pretrain/outputs/$BASE_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation flash_attention_2