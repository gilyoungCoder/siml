#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# JTT Stage 2: Upweighted Training
#
# Usage:
#   ./train_jtt_stage2.sh <step_number>
#   예: ./train_jtt_stage2.sh 1000
#
# Stage 1에서 저장된 checkpoint를 사용하여 Error Set 생성 후 upweight 학습
#───────────────────────────────────────────────────────────────────────────────#

# 선택된 Stage 1 step (인자로 받음)
SELECTED_STEP=${1:-}

if [ -z "$SELECTED_STEP" ]; then
    echo "Usage: $0 <step_number>"
    echo ""
    echo "Available checkpoints:"
    ls -d work_dirs/jtt_stage1/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n
    echo ""
    echo "Example: $0 1000"
    exit 1
fi

STAGE1_CHECKPOINT="work_dirs/jtt_stage1/checkpoints/step_${SELECTED_STEP}/classifier.pth"

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $STAGE1_CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -d work_dirs/jtt_stage1/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n
    exit 1
fi

# (1) GPU 설정
export CUDA_VISIBLE_DEVICES=6

# (2) 데이터 경로
DATA_ROOT=/mnt/home/yhgil99/dataset/threeclassImg/nudity
HARM_DIR="${DATA_ROOT}/harm"
HARM_FAILURE_DIR="${DATA_ROOT}/harm_failure_real"
SAFE_DIR="${DATA_ROOT}/safe"
SAFE_FAILURE_DIR="${DATA_ROOT}/safe_failure"
BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k

# (3) 모델 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="work_dirs/jtt_stage2_T${SELECTED_STEP}"
NUM_CLASSES=3

# (4) JTT 하이퍼파라미터
UPWEIGHT_FACTOR=20.0    # Error set upweight 배수 (논문 권장: 20-100)
MINORITY_RATIO=0.05     # Stage 1과 동일하게 유지

# (5) 학습 설정
NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SAVE_FREQ=100
SEED=42

# (6) WandB 설정
USE_WANDB="--use_wandb --report_to wandb --wandb_project jtt_stage2 --wandb_run_name stage2_T${SELECTED_STEP}_up${UPWEIGHT_FACTOR}"

# (7) 실행
echo "========================================"
echo "JTT Stage 2: Upweighted Training"
echo "========================================"
echo "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "Selected T (step): $SELECTED_STEP"
echo "Upweight factor: $UPWEIGHT_FACTOR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

nohup python train_jtt_stage2.py \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --harm_dir "$HARM_DIR" \
  --harm_failure_dir "$HARM_FAILURE_DIR" \
  --safe_dir "$SAFE_DIR" \
  --safe_failure_dir "$SAFE_FAILURE_DIR" \
  --benign_dir "$BENIGN_DIR" \
  --num_classes $NUM_CLASSES \
  --minority_ratio $MINORITY_RATIO \
  --upweight_factor $UPWEIGHT_FACTOR \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --save_ckpt_freq $SAVE_FREQ \
  --seed $SEED \
  $USE_WANDB \
  > train_jtt_stage2.log 2>&1 &

echo ""
echo "Training launched in background!"
echo "Log file: train_jtt_stage2.log"
echo "Monitor with: tail -f train_jtt_stage2.log"
echo "========================================"
