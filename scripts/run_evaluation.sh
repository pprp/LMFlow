#!/bin/bash

# if [ ! -d data/MedQA-USMLE ]; then
#   cd data && ./download.sh MedQA-USMLE && cd -
# fi

deepspeed_args="--master_port=11011 --include localhost:0"

# CUDA_VISIBLE_DEVICES=1 \
deepspeed ${deepspeed_args} \
    examples/evaluation.py \
    --answer_type text \
    --model_name_or_path /hpc2hdd/JH_DATA/share/xliu886/xliu886_xliu886_share_models/Llama-2-7b-hf \
    --dataset_path data/pretrain_test/ \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 32 \
    --metric ppl


# python  \
#     examples/evaluation.py \
#     --answer_type text \
#     --model_name_or_path /hpc2hdd/JH_DATA/share/xliu886/xliu886_xliu886_share_models/Llama-2-13b-hf \
#     --dataset_path /hpc2hdd/home/pdong212/workspace/LMFlow/data/pretrain_test/ \
#     --inference_batch_size_per_device 8 \
#     --metric ppl
