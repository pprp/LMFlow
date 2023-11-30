#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh

# if [ ! -d data/alpaca ]; then
#   cd data && ./download.sh alpaca && cd -
# fi
deepspeed_args="--master_port=11001 --include localhost:0"


CUDA_VISIBLE_DEVICES=0 \
deepspeed ${deepspeed_args} \
    examples/evaluation.py \
    --answer_type text \
    --model_name_or_path /hpc2hdd/JH_DATA/share/xliu886/xliu886_xliu886_share_models/Llama-2-13b-hf \
    --lora_model_path output_models/pretrain \
    --dataset_path data/pretrain_test \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 16 \
    --metric ppl
