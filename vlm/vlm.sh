#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# IMG_DIR="/mnt/home/yhgil99/unlearning/10_classificaiton/Continual/CountryBodyFixed/10"
IMG_DIR="/mnt/home/yhgil99/unlearning/10_classificaiton/Continual/SDBaseline"

# Python 평가 스크립트 실행
echo "📦 이미지 평가 시작 (디렉토리: $IMG_DIR)"
python vlm.py "$IMG_DIR"
