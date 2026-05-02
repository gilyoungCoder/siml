#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# 15-class 분류기 학습용 실행 스크립트
# (Not People + fully_clothed + casual_wear + summer_casual + athletic_wear +
#  one_piece_swimwear + bikini_swimwear + lingerie + topless_with_jeans +
#  implied_nude + artistic_full_nude +
#  monet_full_style + monet_light_style +
#  vangogh_full_style + vangogh_light_style
#  총 15개 클래스로 DDPM 노이즈 주입 학습)
#───────────────────────────────────────────────────────────────────────────────#

# (1) GPU 지정
export CUDA_VISIBLE_DEVICES=1

# (2) 데이터 루트
DATA_ROOT=/mnt/home/yhgil99/dataset/ArtisticNudity

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
MONET_FULL_STYLE_DIR=$DATA_ROOT/monet_full_style
MONET_LIGHT_STYLE_DIR=$DATA_ROOT/monet_light_style
VANGOGH_FULL_STYLE_DIR=$DATA_ROOT/vangogh_full_style
VANGOGH_LIGHT_STYLE_DIR=$DATA_ROOT/vangogh_light_style

# (4) 모델·출력 설정
PRETRAINED_MODEL=runwayml/stable-diffusion-v1-5
OUTPUT_DIR=work_dirs/fifteen_class_output

# (5) 하이퍼파라미터
BATCH_SIZE=32
LEARNING_RATE=1e-4
SAVE_FREQ=100
MAX_EPOCHS=60

# (6) WandB 설정
USE_WANDB="--use_wandb --report_to wandb --wandb_project fifteen_class_project --wandb_run_name fifteen_class_run"

# (7) 실행
nohup python train_15class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --not_people_data_path        "$NOT_PEOPLE_DIR" \
  --fully_clothed_data_path     "$FULLY_CLOTHED_DIR" \
  --casual_wear_data_path       "$CASUAL_WEAR_DIR" \
  --summer_casual_data_path     "$SUMMER_CASUAL_DIR" \
  --athletic_wear_data_path     "$ATHLETIC_WEAR_DIR" \
  --one_piece_swimwear_path     "$ONE_PIECE_SWIMWEAR_DIR" \
  --bikini_swimwear_path        "$BIKINI_SWIMWEAR_DIR" \
  --lingerie_data_path          "$LINGERIE_DIR" \
  --topless_with_jeans_path     "$TOPLESS_WITH_JEANS_DIR" \
  --implied_nude_data_path      "$IMPLIED_NUDE_DIR" \
  --artistic_full_nude_path     "$ARTISTIC_FULL_NUDE_DIR" \
  --monet_full_style_path       "$MONET_FULL_STYLE_DIR" \
  --monet_light_style_path      "$MONET_LIGHT_STYLE_DIR" \
  --vangogh_full_style_path     "$VANGOGH_FULL_STYLE_DIR" \
  --vangogh_light_style_path    "$VANGOGH_LIGHT_STYLE_DIR" \
  --output_dir                  "$OUTPUT_DIR" \
  --train_batch_size            $BATCH_SIZE \
  --learning_rate               $LEARNING_RATE \
  --save_ckpt_freq              $SAVE_FREQ \
  --num_train_epochs            $MAX_EPOCHS \
  $USE_WANDB \
  > train_15class.log 2>&1 &

echo "Training launched. Logs are in train_15class.log"
