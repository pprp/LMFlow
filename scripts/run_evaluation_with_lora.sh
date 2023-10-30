#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh

# if [ ! -d data/alpaca ]; then
#   cd data && ./download.sh alpaca && cd -
# fi
deepspeed_args="--master_port=11001 --include localhost:4"


# CUDA_VISIBLE_DEVICES=4 \
deepspeed ${deepspeed_args} \
    examples/evaluation.py \
    --answer_type text \
    --model_name_or_path /data/yrb/tmp/Chat-Musician/model/checkpoints/pt_0920/llama2_origin_hf_dir \
    --lora_model_path output_models/finetune \
    --dataset_path data/ \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric ppl
