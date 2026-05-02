#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6

DATA=/mnt/home/yhgil99/dataset/sdxlLight/5class

nohup python train_5class.py \
  --non_people_data_path            "$DATA/imagenet" \
  --clothed_everyday_data_path      "$DATA/clothed_everyday" \
  --clothed_swimwear_data_path      "$DATA/clothed_swimwear" \
  --partial_nude_data_path          "$DATA/nude_partial" \
  --full_nude_data_path             "$DATA/nude_full" \
  --sdxl_base                       stabilityai/stable-diffusion-xl-base-1.0 \
  --output_dir                      ./work_dirs/sdxl1024_hier2 \
  --train_batch_size                8 \
  --learning_rate                   1e-4 \
  --num_train_epochs                10 \
  --use_wandb \
  --wandb_project                   nudity5_hier \
  --wandb_run_name                  run2 \
  > train3.log 2>&1 &

echo "Launched – logs at train1.log"
