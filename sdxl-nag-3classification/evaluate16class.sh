#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# 11-class 분류기 평가용 실행 스크립트
# (Not People + fully_clothed + casual_wear + summer_casual + athletic_wear + \
#  one_piece_swimwear + bikini_swimwear + lingerie + topless_with_jeans + \
#  implied_nude + artistic_full_nude 총 11개 클래스 분류기 평가)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=0

# (2) 데이터 경로 루트 설정
DATA_ROOT=/mnt/home/yhgil99/dataset/ArtisticNudity

# (3) 클래스 디렉토리 경로
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
BENIGN_ARTS=$DATA_ROOT/benignArts

# (4) 평가 관련 설정
PRETRAINED_MODEL=runwayml/stable-diffusion-v1-5
CLASSIFIER_CKPT=/mnt/home/yhgil99/unlearning/15_classificaiton/work_dirs/sixteen_class_output_after4600/checkpoint/step_18000/classifier.pth
# 실제 경로는 사용자의 환경에 맞게 수정하세요.

# (5) 하이퍼파라미터
BATCH_SIZE=16
SEED=42

# (6) 평가 실행
python evaluate16class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --not_people_dir "$NOT_PEOPLE_DIR" \
  --fully_clothed_dir "$FULLY_CLOTHED_DIR" \
  --casual_wear_dir "$CASUAL_WEAR_DIR" \
  --summer_casual_dir "$SUMMER_CASUAL_DIR" \
  --athletic_wear_dir "$ATHLETIC_WEAR_DIR" \
  --one_piece_swimwear_dir "$ONE_PIECE_SWIMWEAR_DIR" \
  --bikini_swimwear_dir "$BIKINI_SWIMWEAR_DIR" \
  --lingerie_dir "$LINGERIE_DIR" \
  --topless_with_jeans_dir "$TOPLESS_WITH_JEANS_DIR" \
  --implied_nude_dir "$IMPLIED_NUDE_DIR" \
  --artistic_full_nude_dir "$ARTISTIC_FULL_NUDE_DIR" \
  --monet_full_style_dir       "$MONET_FULL_STYLE_DIR" \
  --monet_light_style_dir      "$MONET_LIGHT_STYLE_DIR" \
  --vangogh_full_style_dir     "$VANGOGH_FULL_STYLE_DIR" \
  --vangogh_light_style_dir    "$VANGOGH_LIGHT_STYLE_DIR" \
  --benign_arts_dir            "$BENIGN_ARTS" \
  --batch_size $BATCH_SIZE \
  --seed $SEED

echo "11-class classifier evaluation completed."