#!/bin/bash 

# nohup bash scripts/run_finetune_with_lora.sh > lora_finetune.log 2>&1 &
# nohup bash scripts/run_evaluation_with_lora.sh > lora_finetune_test.log 2>&1 &
# nohup bash scripts/run_evaluation_with_lora.sh > lora_pretrain_test_w_lora.log 2>&1 &
# nohup bash scripts/run_evaluation.sh > lora_pretrain_test_wo_lora.log 2>&1 &

# merge lora into weight 
nohup python merge_lora_2.py 