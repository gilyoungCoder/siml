#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

DATA=/mnt/home/yhgil99/dataset/sdxlLight    # not_people / fully_clothed / nude_people

nohup python train_4classadv.py \
  --not_people_data_path    "$DATA/imagenet" \
  --fully_clothed_data_path "$DATA/3class/fullyclothed" \
  --full_nude_data_path   "$DATA/3class/Wnudity" \
  --partial_nude_data_path  "$DATA/10class/lingerie" \
  --sdxl_base stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir work_dirs/multisdxl/lingerie_adv \
  --train_batch_size 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 12 \
  --use_wandb \
  --wandb_project 4class_multi \
  --wandb_run_name run1 \
  > train4.log 2>&1 &
echo "Launched – logs at train1.log"
