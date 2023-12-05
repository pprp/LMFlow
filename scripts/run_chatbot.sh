#!/bin/bash

model="/hpc2hdd/home/pdong212/workspace/share_models/Llama-2-13b-hf"
lora_args="--lora_model_path /hpc2hdd/home/pdong212/workspace/LMFlow/output_models/chatbotv3"
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi

deepspeed_args="--master_port=10221 --include localhost:0"
CUDA_VISIBLE_DEVICES=0 \
  deepspeed ${deepspeed_args} \
  examples/chatbot.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      --max_new_tokens 512 \
      --end_string "###" \
      --prompt_structure "### User:{input_text} ### Agent:" \
      ${lora_args}
