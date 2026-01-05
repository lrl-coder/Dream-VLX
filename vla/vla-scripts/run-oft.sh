export DATA_ROOT_DIR="/path_to/modified_libero_rlds"  #libero_10_no_noops path
export RUN_ROOT_DIR="/path_to/data/vla_output/openvla_libero_10_no_noops" # save train result path                 
export WANDB_API_KEY=

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir $DATA_ROOT_DIR \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir $RUN_ROOT_DIR \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project openvla-finetune \
  --wandb_entity jiacheng-ye \
  --run_id_note parallel_dec--8_acts_chunk_l1--wrist_proprio
