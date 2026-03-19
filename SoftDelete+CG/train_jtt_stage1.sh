#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# JTT Stage 1: Identification Model Training
#
# Train ERM model and save checkpoints at regular intervals.
# After training, check WandB/logs to find optimal T, then run Stage 2.
#───────────────────────────────────────────────────────────────────────────────#

# (1) GPU 설정
export CUDA_VISIBLE_DEVICES=5

# (2) 데이터 경로
DATA_ROOT=/mnt/home/yhgil99/dataset/threeclassImg/
HARM_DIR="${DATA_ROOT}/Wnudity5k"
HARM_FAILURE_DIR="${DATA_ROOT}/Wnudity5k"
SAFE_DIR="${DATA_ROOT}/People5k"
SAFE_FAILURE_DIR="${DATA_ROOT}/People5k"
BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k

# (3) 모델 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR=work_dirs/jtt_wnudity_checking
NUM_CLASSES=3

# (4) JTT 데이터 비율 설정 (중요!)
# 논문에서는 minority가 전체의 ~5% (Waterbirds 기준)
# minority_ratio=0.05 → 95% majority, 5% minority
MINORITY_RATIO=0.05

# (5) 학습 설정
NUM_EPOCHS=150           # 충분히 길게 학습해서 overfitting 시점 확인
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
SEED=42

# (5) 체크포인트/분석 빈도
SAVE_FREQ=100           # 매 100 step마다 checkpoint 저장
ANALYZE_FREQ=500        # 매 500 step마다 error set 분석

# (6) WandB 설정
USE_WANDB="--use_wandb --report_to wandb --wandb_project jtt_stage1 --wandb_run_name stage1_${NUM_CLASSES}class"

# (7) 실행
echo "========================================"
echo "JTT Stage 1: Identification Model"
echo "========================================"
echo "Epochs: $NUM_EPOCHS"
echo "Checkpoint freq: every $SAVE_FREQ steps"
echo "Error set analysis freq: every $ANALYZE_FREQ steps"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

nohup python train_jtt_stage1.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --harm_dir "$HARM_DIR" \
  --harm_failure_dir "$HARM_FAILURE_DIR" \
  --safe_dir "$SAFE_DIR" \
  --safe_failure_dir "$SAFE_FAILURE_DIR" \
  --benign_dir "$BENIGN_DIR" \
  --num_classes $NUM_CLASSES \
  --minority_ratio $MINORITY_RATIO \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $NUM_EPOCHS \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --save_ckpt_freq $SAVE_FREQ \
  --analyze_error_set_freq $ANALYZE_FREQ \
  --seed $SEED \
  $USE_WANDB \
  > train_jtt_stage1.log 2>&1 &

echo ""
echo "Training launched in background!"
echo "Log file: train_jtt_stage1.log"
echo ""
echo "Next steps after training:"
echo "  1. Check WandB or log: tail -f train_jtt_stage1.log"
echo "  2. Find optimal T (step where val acc is good but not overfitting)"
echo "  3. Run Stage 2 with: ./train_jtt_stage2.sh <step_number>"
echo "========================================"
