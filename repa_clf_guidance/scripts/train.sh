#!/usr/bin/env bash
# Train z0 classifier for REPA/SiT guidance
# Uses pretrained SiT-XL/2 (auto-download) + existing 3-class training data
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/repa_clf_guidance

BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"
OUTPUT_DIR="work_dirs/repa_z0_classifier"

nohup python train_z0.py \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_classes 3 \
  --train_batch_size 128 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 20000 \
  --save_ckpt_freq 100 \
  --seed 42 \
  --balance_classes \
  --sit_model "SiT-XL/2" \
  --encoder_depth 8 \
  --resolution 256 \
  --vae_type ema \
  --use_wandb \
  --wandb_project repa_clf_guidance \
  --wandb_run_name z0_resnet18_run1 \
  > logs/train_z0.log 2>&1 &

echo "Training started. PID=$!"
echo "Log: logs/train_z0.log"
