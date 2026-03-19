#!/usr/bin/env bash
# [Z0-IMG-ViT] GPU 0 — nudity 3-class (image space, ViT-B/16 from scratch)
export CUDA_VISIBLE_DEVICES=1

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

nohup python train_img.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --benign_data_path "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
  --person_data_path "/mnt/home/yhgil99/dataset/threeclassImg/clothed5k" \
  --nudity_data_path "/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k" \
  --output_dir "work_dirs/z0_img_vit_classifier" \
  --architecture vit_b \
  --num_classes 3 \
  --train_batch_size 4 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 10000 \
  --save_ckpt_freq 100 \
  --seed 42 \
  --balance_classes \
  --use_wandb \
  --wandb_project z0_img_vit_classifier \
  --wandb_run_name vit_b_img_run1 \
  > logs/train_img_vit_nudity.log 2>&1 &

echo "[GPU 0] nudity 3-class (img, ViT-B) started. PID=$!"
