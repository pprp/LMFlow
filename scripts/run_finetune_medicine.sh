#!/bin/bash

bash ./scripts/run_finetune_with_lora.sh --model_name_or_path /data2/dongpeijie/share/llama-2-7b --dataset_path ./data/PubMedQA/train  --output_lora_path output_models/finetuned_pubmedqa