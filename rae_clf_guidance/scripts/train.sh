#!/usr/bin/env bash
# Train z0 classifier for RAE/DiTDH guidance
# Uses pretrained DiTDH-XL + DINOv2 encoder + existing 3-class training data
export CUDA_VISIBLE_DEVICES=1

cd /mnt/home/yhgil99/unlearning/rae_clf_guidance

BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"
OUTPUT_DIR="work_dirs/rae_z0_classifier"

# Pretrained model paths (created by setup.sh)
DITDH_CKPT="pretrained_models/stage2_model.pt"
STAT_PATH="pretrained_models/stat.pt"
DECODER_CFG="rae_src/configs/decoder/ViTXL"

mkdir -p logs

nohup python train_z0.py \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_classes 3 \
  --train_batch_size 4 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 20000 \
  --save_ckpt_freq 100 \
  --seed 42 \
  --mixed_precision bf16 \
  --balance_classes \
  --ditdh_ckpt "$DITDH_CKPT" \
  --stat_path "$STAT_PATH" \
  --decoder_config_path "$DECODER_CFG" \
  --use_time_shift \
  --use_wandb \
  --wandb_project rae_clf_guidance \
  --wandb_run_name z0_dinov2_mlp_run1 \
  > logs/train_z0.log 2>&1 &

echo "Training started. PID=$!"
echo "Log: logs/train_z0.log"
