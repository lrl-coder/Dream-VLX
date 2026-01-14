#!/bin/bash

CUDA_VISIBLE_DEVICES=0
total_tasks=50
num_gpus=1
path="./checkpoint/fine-tuned"  # PATH to finetuned dream-vla ckpt

log_dir=$path/eval_logs
mkdir -p $log_dir

for ((j=0; j<total_tasks; j++)); do
    gpu_idx=$(( j % num_gpus ))
        
    cmd="CUDA_VISIBLE_DEVICES=${gpu_idx} python experiments/robot/libero/run_libero_eval_dream.py \
        --pretrained_checkpoint $path \
        --task_suite_name libero_10 \
        --center_crop True \
        --episode_idx $j \
        --num_trials_per_task 1 \
        --rollout_dir $path/rollouts \
        --local_log_dir $path/evals \
        --use_flow_matching True \
        --num_images_in_input 2 \
        --action_chunk 8 \
        --steps 4 \
        --num_open_loop_steps 8 \
        --use_proprio True \
        --run_id_note trial_${j} > $log_dir/trial_${j}.log 2>&1 &"
    echo $cmd
    eval $cmd
    
        
    if (( (j+1) % num_gpus == 0 )); then
        wait
    fi
done

python vla-scripts/cal_acc.py $path/evals