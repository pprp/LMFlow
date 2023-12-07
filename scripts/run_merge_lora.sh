# https://wandb.ai/pprp/huggingface/overview
#!/bin/bash 

python examples/merge_lora.py \
    --model_name_or_path /hpc2hdd/JH_DATA/share/xliu886/xliu886_xliu886_share_models/Llama-2-13b-hf \
    --lora_model_path ./output_models/chatbotv2 \
    --output_model_path ./output_models/merged_lora_chatbotv2