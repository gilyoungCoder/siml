#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────
# evaluate_single_image.sh
# 학습된 3-클래스 분류기로 폴더 내 모든 이미지를 평가하는 스크립트
#───────────────────────────────────────────────────────────────────────────────#

# 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=1
# DATA_ROOT=./nullguidance

DATA_ROOT=/mnt/home/yhgil99/dataset/sdxlLight/cheatingData1


# 사전학습된 VAE & Scheduler 체크포인트
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"

# 훈련된 분류기 체크포인트 경로
CLASSIFIER_CKPT="./work_dirs/sdxl1024/classifier_final.pth"

# 평가할 이미지가 들어있는 폴더
# IMAGE_DIR="./Continual/CountryNudeBodyPure/15"
IMAGE_DIR=$DATA_ROOT
python checkClass.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --image_dir "$IMAGE_DIR" \
    --batch_size 16
