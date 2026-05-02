#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# three_class 분류기 학습용 실행 스크립트 (Balanced Color Augmentation 적용)
# (benign / person / nudity 3개 클래스로 DDPM 노이즈 주입 학습)
# - BalancedColorAugmentation: 피부톤 의존성 제거를 위한 적정 수준 color augmentation
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=5

# (2) 학습에 사용할 경로 설정
BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k        # 사람 없음
PERSON_DIR=/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe     # 사람 있음(비누드)
NUDITY_DIR=/mnt/home/yhgil99/dataset/threeclassImg/nudity/harm       # 사람 누드

# (3) 모델/출력 관련 설정
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"       # 사용할 VAE/scheduler
OUTPUT_DIR=work_dirs/nudity_three_class_color_first   # 체크포인트·로그 저장 폴더

# (4) 하이퍼파라미터
BATCH_SIZE=40
LEARNING_RATE=1e-4
SAVE_FREQ=100                                           # 스텝마다 체크포인트 저장
MAX_EPOCHS=150                                           # --num_train_epochs
# MIXED_PRECISION="fp16"                                  # no | fp16 | bf16
#   --mixed_precision $MIXED_PRECISION \

# (5) WandB 설정 (사용 안 하려면 --use_wandb, --report_to 옵션 제외)
USE_WANDB="--use_wandb --report_to wandb --wandb_project three_class_project_color --wandb_run_name threeclass_color_first"

# (6) 스크립트 실행
nohup python train_3class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  $USE_WANDB \
  > trainClass_color_aug.log 2>&1 &

echo "Training launched with Balanced Color Augmentation. Logs are in trainClass_color_aug.log"
