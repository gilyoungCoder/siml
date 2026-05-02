#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# 11-class 분류기 학습용 실행 스크립트
# (Not People + fully_clothed + … + artistic_full_nude 총 11개 클래스로 DDPM 노이즈 주입 학습)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=0

# (2) 데이터 경로 루트 설정
DATA_ROOT=/mnt/home/yhgil99/dataset/10classesNudity

# (3) 클래스별 디렉토리
NOT_PEOPLE_DIR=$DATA_ROOT/imagenet5k
FULLY_CLOTHED_DIR=$DATA_ROOT/fully_clothed
CASUAL_WEAR_DIR=$DATA_ROOT/casual_wear
SUMMER_CASUAL_DIR=$DATA_ROOT/summer_casual
ATHLETIC_WEAR_DIR=$DATA_ROOT/athletic_wear
ONE_PIECE_SWIMWEAR_DIR=$DATA_ROOT/one_piece_swimwear
BIKINI_SWIMWEAR_DIR=$DATA_ROOT/bikini_swimwear
LINGERIE_DIR=$DATA_ROOT/lingerie
TOPLESS_WITH_JEANS_DIR=$DATA_ROOT/topless_with_jeans
IMPLIED_NUDE_DIR=$DATA_ROOT/implied_nude
ARTISTIC_FULL_NUDE_DIR=$DATA_ROOT/artistic_full_nude

# (4) 모델·출력 관련 설정
PRETRAINED_MODEL=runwayml/stable-diffusion-v1-5
OUTPUT_DIR=work_dirs/eleven_class_output

# (5) 하이퍼파라미터
BATCH_SIZE=32
LEARNING_RATE=1e-4
SAVE_FREQ=100           # 스텝마다 체크포인트 저장
MAX_EPOCHS=60           # --num_train_epochs
# MIXED_PRECISION="fp16" # no | fp16 | bf16

# (6) WandB 설정 (미사용 시 빈 문자열로)
USE_WANDB="--use_wandb --report_to wandb --wandb_project eleven_class_project --wandb_run_name eleven_class_run"

# (7) 학습 실행
nohup python train_11class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --not_people_data_path       "$NOT_PEOPLE_DIR" \
  --fully_clothed_data_path    "$FULLY_CLOTHED_DIR" \
  --casual_wear_data_path      "$CASUAL_WEAR_DIR" \
  --summer_casual_data_path    "$SUMMER_CASUAL_DIR" \
  --athletic_wear_data_path    "$ATHLETIC_WEAR_DIR" \
  --one_piece_swimwear_path    "$ONE_PIECE_SWIMWEAR_DIR" \
  --bikini_swimwear_path       "$BIKINI_SWIMWEAR_DIR" \
  --lingerie_data_path         "$LINGERIE_DIR" \
  --topless_with_jeans_path    "$TOPLESS_WITH_JEANS_DIR" \
  --implied_nude_data_path     "$IMPLIED_NUDE_DIR" \
  --artistic_full_nude_path    "$ARTISTIC_FULL_NUDE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --save_ckpt_freq $SAVE_FREQ \
  --num_train_epochs $MAX_EPOCHS \
  $USE_WANDB \
  > run.log 2>&1 &

echo "Training launched. Logs are in run.log"
