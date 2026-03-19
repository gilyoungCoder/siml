#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────
# evaluate_single_image.sh
# 학습된 3-클래스 분류기로 폴더 내 모든 이미지를 평가하는 스크립트
#───────────────────────────────────────────────────────────────────────────────#

# 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=0

# 사전학습된 VAE & Scheduler 체크포인트
PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"

# 훈련된 분류기 체크포인트 경로
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/three_classificaiton_new/work_dirs/nudity_three_class_Imagenet_Filtered/checkpoint/step_21800/classifier.pth"

# 평가할 이미지가 들어있는 폴더
IMAGE_DIR="Continual/CountryBody2People/allPNDM/45"
IMG_PATH="$IMAGE_DIR/12_1.png"

python checkClass.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --image "$IMG_PATH"
