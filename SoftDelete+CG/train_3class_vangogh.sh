#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# three_class 분류기 학습용 실행 스크립트 (Van Gogh Edition)
# (benign / vangogh_anti / vangogh 3개 클래스로 DDPM 노이즈 주입 학습)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 학습에 사용할 경로 설정
export CUDA_VISIBLE_DEVICES=5

# (2) 학습에 사용할 경로 설정
BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k           # 폭력 없음
PERSON_DIR=/mnt/home/yhgil99/dataset/threeclassImg/VanGogh/safe      # 반폭력 (평화적 이미지)
VIOLENCE_DIR=/mnt/home/yhgil99/dataset/threeclassImg/VanGogh/harm           # Van Gogh 스타일

# (3) 모델/출력 관련 설정
PRETRAINED_MODEL=CompVis/stable-diffusion-v1-4        # 사용할 VAE/scheduler
OUTPUT_DIR=work_dirs/vangogh_three_class_diff_new              # 체크포인트·로그 저장 폴더

# (4) 하이퍼파라미터
BATCH_SIZE=40
LEARNING_RATE=1e-4
SAVE_FREQ=500                                           # 스텝마다 체크포인트 저장
MAX_EPOCHS=300                                           # --num_train_epochs
# MIXED_PRECISION="fp16"                                  # no | fp16 | bf16
#   --mixed_precision $MIXED_PRECISION \

# (5) WandB 설정 (사용 안 하려면 --use_wandb, --report_to 옵션 제외)
USE_WANDB="--use_wandb --report_to wandb --wandb_project vangogh_three_class_project --wandb_run_name vangogh_3class_run"

# (6) 스크립트 실행
nohup python train_3class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$VIOLENCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  $USE_WANDB \
  > train_vangogh_3class.log 2>&1 &

echo "Van Gogh 3-class training launched. Logs are in train_vangogh_3class.log"
