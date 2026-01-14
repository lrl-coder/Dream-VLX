export DATA_ROOT_DIR="modified_libero_rlds"  # modified_libero_rlds path
export WANDB_API_KEY="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"
export CUDA_VISIBLE_DEVICES=1,2
for task in 10 goal object spatial; do
  export RUN_ROOT_DIR="./exp/vla/libero_${task}_no_noops" # save train result path                 
  

  torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_dream.py \
    --vla_path ./checkpoint \
    --data_root_dir $DATA_ROOT_DIR \
    --dataset_name libero_${task}_no_noops \
    --run_root_dir $RUN_ROOT_DIR \
    --use_l1_regression False \
    --use_diffusion False \
    --use_flow_matching True \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 200005 \
    --save_freq 50000 \
    --save_latest_checkpoint_only True \
    --image_aug True \
    --lora_rank 8 \
    --wandb_project openvla-finetune \
    --wandb_entity 1559589961-northwestern-university \
    --run_id_note parallel_dec--8_acts_chunk_flow_matching--wrist_proprio
done