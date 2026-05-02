#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5

DATA=/mnt/home/yhgil99/dataset/sdxlLight    # not_people / fully_clothed / nude_people

nohup python train_3class.py \
  --not_people_data_path    "$DATA/imagenet" \
  --fully_clothed_data_path "$DATA/3class/fullyclothed" \
  --nude_people_data_path   "$DATA/3class/Wnudity" \
  --sdxl_base stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir work_dirs/sdxl1024_after1000 \
  --train_batch_size 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --use_wandb \
  --report_to wandb \
  --wandb_project nudity2class \
  --wandb_run_name run1 \
  > train.log 2>&1 &
echo "Launched – logs at train.log"
