#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# 5-class 분류기 학습용 실행 스크립트
# (Nonpeople + Clothed + Revealing (Non-nude) + Partial Nudity + Full Nudity 총 5개 클래스로 DDPM 노이즈 주입 학습)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=7

# (2) 데이터 경로 루트 설정
DATA_ROOT=/mnt/home/yhgil99/dataset/5classNudity/vanila

# (3) 클래스별 디렉터리
NOT_PEOPLE_DIR=$DATA_ROOT/imagenet5k
CLOTHED_DIR=$DATA_ROOT/Clothed
REVEALING_DIR=$DATA_ROOT/Revealing_Non_nude
FULL_NUDITY_DIR=$DATA_ROOT/Full_Nudity

# (4) 모델·출력 관련 설정
PRETRAINED_MODEL=CompVis/stable-diffusion-v1-4
OUTPUT_DIR=work_dirs/five_class_output

# (5) 하이퍼파라미터
BATCH_SIZE=16
LEARNING_RATE=1e-4
SAVE_FREQ=100           # 스텝마다 체크포인트 저장
MAX_EPOCHS=30           # --num_train_epochs
# MIXED_PRECISION="fp16" # no | fp16 | bf16

# (6) WandB 설정 (미사용 시 빈 문자열로)
USE_WANDB="--use_wandb --report_to wandb --wandb_project four_class_project --wandb_run_name four_class_run"

# (7) 학습 실행
nohup python train_4class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --nonpeople_dir "$NOT_PEOPLE_DIR" \
  --clothed_dir "$CLOTHED_DIR" \
  --revealing_dir "$REVEALING_DIR" \
  --full_nudity_dir "$FULL_NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  $USE_WANDB \
  > run4class.log 2>&1 &

echo "5-class Training launched. Logs are in run5class.log"

