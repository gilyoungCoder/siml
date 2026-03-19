#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# Full JTT (Just Train Twice) Training Pipeline
#
# Stage 1: Train ERM model for T epochs → Find Error Set (misclassified samples)
# Stage 2: Train new model with Error Set upweighted by λ_up
#
# This prevents skin-tone dependency in nudity classification
#───────────────────────────────────────────────────────────────────────────────#

# (1) GPU 설정
export CUDA_VISIBLE_DEVICES=6

# (2) 데이터 경로 설정
DATA_ROOT=/mnt/home/yhgil99/dataset/threeclassImg/nudity
HARM_DIR="${DATA_ROOT}/harm"                    # nude + 원래 피부톤
HARM_FAILURE_DIR="${DATA_ROOT}/harm_failure"    # nude + 파란 피부톤
SAFE_DIR="${DATA_ROOT}/safe"                    # clothed person
SAFE_FAILURE_DIR="${DATA_ROOT}/safe_failure"    # beige colored clothed

# 3-class용 benign 데이터 (옵션)
BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k

# (3) 모델 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR=work_dirs/nudity_jtt_full
NUM_CLASSES=3  # 2: harm/safe, 3: benign/safe/harm

# (4) JTT 하이퍼파라미터 (중요!)
STAGE1_EPOCHS=10      # Stage 1 학습 epochs (T) - 논문 권장: 전체의 40-60%
STAGE2_EPOCHS=30      # Stage 2 학습 epochs
UPWEIGHT_FACTOR=20.0  # Error set upweight 배수 (λ_up) - 논문 권장: 20-100

# (5) 학습 하이퍼파라미터
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01     # L2 regularization (JTT에서 중요)
SAVE_FREQ=100
SEED=42

# (6) WandB 설정
USE_WANDB="--use_wandb --report_to wandb --wandb_project jtt_full_nudity --wandb_run_name jtt_T${STAGE1_EPOCHS}_up${UPWEIGHT_FACTOR}"

# (7) 실행
echo "========================================"
echo "Full JTT Training Pipeline"
echo "========================================"
echo "Stage 1 epochs (T): $STAGE1_EPOCHS"
echo "Stage 2 epochs: $STAGE2_EPOCHS"
echo "Upweight factor (λ_up): $UPWEIGHT_FACTOR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

nohup python train_3class_jtt_full.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --harm_dir "$HARM_DIR" \
  --harm_failure_dir "$HARM_FAILURE_DIR" \
  --safe_dir "$SAFE_DIR" \
  --safe_failure_dir "$SAFE_FAILURE_DIR" \
  --num_classes $NUM_CLASSES \
  --stage1_epochs $STAGE1_EPOCHS \
  --stage2_epochs $STAGE2_EPOCHS \
  --upweight_factor $UPWEIGHT_FACTOR \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --save_ckpt_freq $SAVE_FREQ \
  --seed $SEED \
  $USE_WANDB \
  > train_jtt_full.log 2>&1 &

echo ""
echo "Training launched in background!"
echo "Log file: train_jtt_full.log"
echo "Monitor with: tail -f train_jtt_full.log"
echo "========================================"
