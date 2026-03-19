#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

DATA=/mnt/home/yhgil99/dataset/sdxlLight    # not_people / fully_clothed / nude_people

nohup python train_4classCheating.py \
  --not_people_data_path    "$DATA/4class/imagenet" \
  --fully_clothed_data_path "$DATA/4class/fullyclothed" \
  --full_nude_data_path   "$DATA/4class/fullnude_r2" \
  --partial_nude_data_path  "$DATA/4class/partial_nude" \
  --sdxl_base stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir work_dirs/cheatingRound/r2 \
  --train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 6 \
  --use_wandb \
  --wandb_project 4class_cheated \
  --wandb_run_name round2 \
  > train6.log 2>&1 &
echo "Launched – logs at train1.log"
