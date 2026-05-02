#!/bin/bash
# run_clipscore.sh
# 이 스크립트는 CLIPScore를 계산하는 파이썬 스크립트를 실행합니다.
# 사용 예시:
# 1. 이미지 폴더: ./images
# 2. 프롬프트 파일: ./images/prompts.txt (이미지 폴더 내에 위치하는 경우)
# 스크립트를 실행하려면: chmod +x run_clipscore.sh && ./run_clipscore.sh

# 파라미터 설정 (필요에 따라 수정)
IMG_PATH="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_nudity/gs7.0_hs1.5_st0.3_ws3.0-0.5_ts-0.5--2.5"
# PROMPTS_PATH를 지정하지 않으면 IMG_PATH/prompts.txt를 자동으로 사용
PROMPTS_PATH="./prompts/sexual_50.txt"  # 비워두면 IMG_PATH/prompts.txt 사용
BATCH_SIZE=32
DEVICE="cuda:7"
PRETRAINED_MODEL="openai/clip-vit-large-patch14"
WEIGHT=1.0
EXT="png"

# 실행할 파이썬 파일 이름 (파이썬 스크립트 파일 이름을 맞춰주세요)
PYTHON_FILE="evaluate/eval_clipscore.py"

# 파이썬 파일이 존재하는지 확인
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Error: $PYTHON_FILE 파일을 찾을 수 없습니다."
    exit 1
fi

# 파이썬 스크립트 실행
if [ -z "$PROMPTS_PATH" ]; then
  # PROMPTS_PATH가 비어있으면 --prompts_path 옵션 없이 실행 (자동으로 IMG_PATH/prompts.txt 사용)
  python "$PYTHON_FILE" \
    --img_path "$IMG_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --w "$WEIGHT" \
    --ext "$EXT"
else
  python "$PYTHON_FILE" \
    --img_path "$IMG_PATH" \
    --prompts_path "$PROMPTS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --w "$WEIGHT" \
    --ext "$EXT"
fi
