#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=/hpc2hdd/home/pdong212/workspace/share_models/Llama-2-7b-hf
dataset_path=data/finetune
output_dir=output_models/finetune_chatbotv4
deepspeed_args="--master_port=11010  --include localhost:0,1"

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune_with_lora_chatbotv4
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --use_lora 1 \
    --lora_r 8 \
    --save_aggregated_lora 1\
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
