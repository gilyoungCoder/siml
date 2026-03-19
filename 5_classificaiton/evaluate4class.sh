#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# 4-class 분류기 평가용 실행 스크립트
# (Nonpeople + Clothed + Revealing + Full Nudity 총 4개 클래스 분류기 평가)
#───────────────────────────────────────────────────────────────────────────────#

# (1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=4

# (2) 데이터 경로 루트 설정
#   └ 아래 예시는 폴더 구조:
#       $DATA_ROOT/
#         ├── Nonpeople/
#         ├── Clothed/
#         ├── Revealing/
#         └── Full_Nudity/
DATA_ROOT=/mnt/home/yhgil99/dataset/5classNudity/vanila

# (3) 클래스별 디렉터리
NONPEOPLE_DIR=$DATA_ROOT/imagenet5k
CLOTHED_DIR=$DATA_ROOT/Clothed
REVEALING_DIR=$DATA_ROOT/Revealing_Non_nude
FULL_NUDITY_DIR=$DATA_ROOT/Full_Nudity

# (4) 평가 관련 설정
PRETRAINED_MODEL=runwayml/stable-diffusion-v1-5
CLASSIFIER_CKPT=/mnt/home/yhgil99/unlearning/5_classificaiton/work_dirs/five_class_output/checkpoint/step_20100/classifier.pth
# ↑ 실제 체크포인트 경로로 수정하세요.

# (5) 하이퍼파라미터
BATCH_SIZE=32
SEED=42

# (6) 평가 실행
python evaluate4class.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --nonpeople_dir "$NONPEOPLE_DIR" \
  --clothed_dir "$CLOTHED_DIR" \
  --revealing_dir "$REVEALING_DIR" \
  --full_nudity_dir "$FULL_NUDITY_DIR" \
  --batch_size $BATCH_SIZE \
  --seed $SEED

echo "4-class classifier evaluation completed."
