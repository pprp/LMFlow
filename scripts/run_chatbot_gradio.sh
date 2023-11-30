#!/bin/bash 

model="output_models/merged_lora"

deepspeed --master_port=11005 examples/chatbot_gradio.py \
    --model_name_or_path ${model} \
    --deepspeed configs/ds_config_chatbot.json \
    --prompt_structure "### User: {input_text}### Agent:" \
    --max_new_tokens 1024