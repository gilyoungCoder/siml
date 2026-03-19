#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────
# evaluate_guidance_results.sh
# partial_adv / partial_multi 의 5,10,15,20,25 폴더를 일괄 평가
#───────────────────────────────────────────────────────────────────────────────

# 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=7

# VAE & scheduler 체크포인트
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
# 4-class 분류기 체크포인트
CLASSIFIER_CKPT="./work_dirs/sdxl1024/classifier_final.pth"
# 결과가 저장된 최상위 폴더
BASE_DIR="/mnt/home/yhgil99/unlearning/sdxl-lightening-4classification/sdxl_guidance_out/CountryNudeBody"

for MODE in partial_adv partial_multi; do
  for SCALE in 5 10 15 20 25; do
    IMAGE_DIR="${BASE_DIR}/${MODE}/${SCALE}"
    echo "▶ 평가 시작: MODE=${MODE}, SCALE=${SCALE}"
    python checkClass.py \
      --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
      --classifier_ckpt         "$CLASSIFIER_CKPT" \
      --image_dir               "$IMAGE_DIR" \
      --batch_size              16
    echo "──────────────────────────────────────────────────────────"
  done
done

echo "✅ 모든 평가가 완료되었습니다."
