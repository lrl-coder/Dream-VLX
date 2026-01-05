## Use chatgpt to extract answers for math tasks
# export OPENAI_API_KEY=
# export OPENAI_API_URL=
module load cuda12.2/toolkit/12.2.2 cuda12.2/nsight/12.2.2 cuda12.2/profiler/12.2.2
CHECKPOINT=Dream-org/Dream-VL-7B
tasks="mmmu_val mmmu_pro_vision mmstar mmbench_en_test seedbench ai2d chartqa infovqa_test docvqa_test realworldqa mathverse_testmini_vision mathvista_testmini_format"
lengths="3 3 8 16 3 3 16 32 32 16 128 128"
steps="1 1 8 16 1 1 16 32 32 16 128 128"

read -ra TASKS_ARRAY <<< "$tasks"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra STEPS_ARRAY <<< "$steps"

for i in "${!TASKS_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port 29591 -m lmms_eval --model dream_vl \
        --model_args pretrained=${CHECKPOINT} \
        --tasks ${TASKS_ARRAY[$i]} \
        --gen_kwargs "max_new_tokens=${LENGTH_ARRAY[$i]},steps=${STEPS_ARRAY[$i]},alg=maskgit_plus" \
        --batch_size 1 \
        --log_samples \
        --output_path results/${TASKS_ARRAY[$i]}
done
