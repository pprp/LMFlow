#!/bin/bash 

# FOR 13B 
# nohup bash scripts/run_finetune_with_lora.sh > lora_finetune.log 2>&1 &
# nohup bash scripts/run_evaluation_with_lora.sh > lora_finetune_test.log 2>&1 &
# nohup bash scripts/run_evaluation_with_lora.sh > lora_pretrain_test_w_lora.log 2>&1 &
# nohup bash scripts/run_evaluation.sh > lora_pretrain_test_wo_lora.log 2>&1 &
# nohup bash scripts/run_finetune_with_lora.sh > lora_chatbotv3.log 2>&1 &

# FOR 7B
# nohup bash scripts/run_evaluation.sh > lora_pretrain_test_w_lora_7b.log 2>&1 &
# nohup bash scripts/run_finetune_with_lora.sh > lora_finetune_7b.log 2>&1 &


# INFERENCE 
bash scripts/run_chatbot.sh
