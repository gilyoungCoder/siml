#!/usr/bin/env bash
# [Z0] GPU 0 — nudity 3-class
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"
OUTPUT_DIR="work_dirs/z0_resnet18_classifier"

nohup python train.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_classes 3 \
  --train_batch_size 16 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10000 \
  --save_ckpt_freq 100 \
  --seed 42 \
  --balance_classes \
  --use_wandb \
  --wandb_project z0_classifier \
  --wandb_run_name resnet18_run1 \
  > logs/train_nudity.log 2>&1 &

echo "[GPU 0] nudity 3-class started. PID=$!"
