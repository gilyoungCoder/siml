#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# JTT (Just Train Twice) 학습 스크립트
# 피부톤 의존성을 줄이기 위해 failure case를 upweight하여 학습
#
# Dataset structure:
#   - harm: nude 이미지 (majority - ERM이 잘 맞추는 것)
#   - harm_failure: nude인데 ERM이 틀린 것 (minority - 어두운 피부톤 등)
#   - safe: safe 이미지 (majority - ERM이 잘 맞추는 것)
#   - safe_failure: safe인데 ERM이 틀린 것 (minority - 밝은 피부톤 등)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=6

# (2) 데이터 경로 설정
DATA_ROOT=/mnt/home/yhgil99/dataset/threeclassImg/nudity
HARM_DIR="${DATA_ROOT}/harm"
HARM_FAILURE_DIR="${DATA_ROOT}/harm_failure_real"
SAFE_DIR="${DATA_ROOT}/safe"
SAFE_FAILURE_DIR="${DATA_ROOT}/safe_failure"

# (3) 모델/출력 관련 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR=work_dirs/nudity_jtt

# (4) JTT 하이퍼파라미터
UPWEIGHT_FACTOR=20.0    # failure case를 몇 배로 upweight할지 (논문 권장: 20~100)

# (5) 학습 하이퍼파라미터
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01       # L2 regularization (JTT에서 중요)
SAVE_FREQ=100
MAX_EPOCHS=45
SEED=42

# (6) WandB 설정
USE_WANDB="--use_wandb --report_to wandb --wandb_project jtt_nudity_classifier --wandb_run_name jtt_up${UPWEIGHT_FACTOR}"

# (7) 스크립트 실행
nohup python train_3class_jtt.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --harm_dir "$HARM_DIR" \
  --harm_failure_dir "$HARM_FAILURE_DIR" \
  --safe_dir "$SAFE_DIR" \
  --safe_failure_dir "$SAFE_FAILURE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --upweight_factor $UPWEIGHT_FACTOR \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  --seed $SEED \
  --use_weighted_sampler \
  --balance_classes \
  $USE_WANDB \
  > train_jtt.log 2>&1 &

echo "========================================"
echo "JTT Training launched!"
echo "========================================"
echo "Output dir: $OUTPUT_DIR"
echo "Upweight factor: $UPWEIGHT_FACTOR"
echo "Log file: train_jtt.log"
echo ""
echo "Monitor with: tail -f train_jtt.log"
echo "========================================"
