#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# two_class 분류기 학습용 실행 스크립트
# (allow=non-nude / harm=nude 2개 클래스로 DDPM 노이즈 주입 학습)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=4

# (2) 원천 데이터 경로 (기존)
      # 누드

# (2-1) allow 합본 폴더 생성(심볼릭 링크)
ALLOW_DIR=/mnt/home/yhgil99/dataset/softDeleteSD/allowlist
HARM_DIR=/mnt/home/yhgil99/dataset/softDeleteSD/harmlist

# 이미지 확장자만 링크(중복 파일명은 마지막 것으로 링크 갱신)

# (3) 모델/출력 관련 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"      # 사용할 VAE/scheduler
OUTPUT_DIR=work_dirs/nudity_two_classSD                 # 체크포인트·로그 저장 폴더
LOG_FILE=train_two_class.log

# (4) 하이퍼파라미터
BATCH_SIZE=32
LEARNING_RATE=1e-4
SAVE_FREQ=100                                         # 스텝마다 체크포인트 저장
MAX_EPOCHS=30                                         # --num_train_epochs
# MIXED_PRECISION="fp16"                              # no | fp16 | bf16
#   --mixed_precision $MIXED_PRECISION \

# (5) Weights & Biases (원하면 비활성화)
USE_WANDB="--use_wandb --report_to wandb --wandb_project two_class_project --wandb_run_name allow_harm_run"

# (6) 스크립트 실행
nohup python train_2class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --allow_data_path "$ALLOW_DIR" \
  --harm_data_path "$HARM_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  $USE_WANDB \
  > "$LOG_FILE" 2>&1 &

echo "Training launched. Logs are in $LOG_FILE"
