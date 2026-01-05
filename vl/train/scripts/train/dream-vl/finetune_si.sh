export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=enp
export NCCL_DEBUG=INFO
# export USE_SYSTEM_NCCL=1
# export NCCL_IB_TIMEOUT=22

LLM_VERSION="Dream-org/Dream-v0-Instruct-7B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="hf:Emova-ollm/qwen2vit600m"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
# cd your_path/LLaVA-NeXT
# source your_path/miniconda3/bin/activate llava
# export HF_HOME=xxx

export WANDB_API_KEY=
export WANDB_PROJECT=Dream-VL

PROMPT_VERSION="qwen_2_5"

BASE_RUN_NAME="dream-vl_sft_si_lr1e-5"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

module load cuda12.2/toolkit/12.2.2 cuda12.2/nsight/12.2.2 cuda12.2/profiler/12.2.2
# module load cuda12.1

NUM_GPUS=8
NNODES=$SLURM_NNODES
WORLD_SIZE=$((NNODES * NUM_GPUS))
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=8 --nnodes=1 --master_port=25901\
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${NUM_GPUS} --nnodes=${NNODES} \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path scripts/train/dream-vl/single_image.yaml \
    --image_folder /path_to/LLaVA-SFT/image_data \
    --pretrain_mm_mlp_adapter=/path_to/LLaVA-Pretrain/outputs/dream-vl_pretrain_qwen2vit/mm_projector.bin \
    --mm_tunable_parts=mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --vision_custom_tunable_parts \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio native_anyres \
    --image_grid_pinpoints '(1x1),...,(6x6)' \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir /path_to/LLaVA-SFT/outputs-si/$BASE_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2