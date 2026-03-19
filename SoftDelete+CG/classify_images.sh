#!/bin/bash
# classify_images.sh
# 임의의 이미지 디렉토리를 받아 분류 수행

export CUDA_VISIBLE_DEVICES=7

# ============================================
# 여기에 이미지 디렉토리 경로를 입력하세요
# ============================================
IMAGE_DIR="./img/CNBWON"

# 출력 파일 경로 (선택사항, 비워두면 자동 생성)
OUTPUT_JSON=""  # 예: "results.json"
OUTPUT_CSV=""   # 예: "results.csv"

# 기본 설정
CLASSIFIER_CKPT=./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth
VAE_MODEL="CompVis/stable-diffusion-v1-4"

# ============================================
# 실행 부분 (수정 불필요)
# ============================================

# 디렉토리 존재 확인
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory not found: $IMAGE_DIR"
    echo "Please edit IMAGE_DIR in this script."
    exit 1
fi

# Python 스크립트 실행
CMD="python classify_images.py \"$IMAGE_DIR\" \
    --classifier_ckpt $CLASSIFIER_CKPT \
    --vae_model $VAE_MODEL \
    --num_classes 3 \
    --batch_size 8 \
    --device cuda \
    --summary \
    --verbose"

# 옵션 추가
if [ -n "$OUTPUT_JSON" ]; then
    CMD="$CMD --output_json \"$OUTPUT_JSON\""
fi

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --output_csv \"$OUTPUT_CSV\""
fi

echo "Running classification..."
echo "Image directory: $IMAGE_DIR"
echo "Command: $CMD"
echo ""

eval $CMD
