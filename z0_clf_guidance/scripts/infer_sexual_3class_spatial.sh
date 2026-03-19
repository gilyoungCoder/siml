#!/usr/bin/env bash
# ============================================================================
# [INFERENCE] 3-class nudity clf + SPATIAL guidance on sexual prompts
# Classifier: ResNet18 (3-class: 0=non-people, 1=clothed, 2=nude)
# Guidance: safe_minus_harm + spatial => safe=[1], harm=[2]
# Fixed guidance_scale=10, sweep spatial_threshold: 0.2, 0.3, 0.5
# Both latent z0 and image space, sequential on one GPU
# ============================================================================
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
mkdir -p logs

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/sexual_high_tox.txt"

Z0_CLF="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
IMG_CLF="./work_dirs/z0_img_resnet18_classifier/checkpoint/step_18900/classifier.pth"

GUIDE_SCALE=10.0
THRESHOLDS="0.2 0.3 0.5"

for thresh in ${THRESHOLDS}; do
    tag=$(echo "${thresh}" | sed 's/\.//g')

    echo "=============================================="
    echo "[3-class sexual spatial] threshold=${thresh}  scale=${GUIDE_SCALE}  (tag=t${tag})"
    echo "=============================================="

    # --- latent z0 ---
    echo "  [z0] starting..."
    python generate.py "${CKPT_PATH}" \
      --prompt_file "${PROMPT_FILE}" \
      --output_dir "./output_img/sexual_z0_3class_spatial_t${tag}" \
      --nsamples 1 --cfg_scale 7.5 --num_inference_steps 50 --seed 1234 \
      --classifier_ckpt "${Z0_CLF}" \
      --architecture resnet18 --num_classes 3 --space latent \
      --guidance_scale "${GUIDE_SCALE}" --guidance_start_step 1 \
      --guidance_mode safe_minus_harm --safe_classes 1 --harm_classes 2 \
      --spatial_guidance --spatial_threshold "${thresh}"

    echo "  [z0] done."

    # --- image space ---
    echo "  [img] starting..."
    python generate.py "${CKPT_PATH}" \
      --prompt_file "${PROMPT_FILE}" \
      --output_dir "./output_img/sexual_img_3class_spatial_t${tag}" \
      --nsamples 1 --cfg_scale 7.5 --num_inference_steps 50 --seed 1234 \
      --classifier_ckpt "${IMG_CLF}" \
      --architecture resnet18 --num_classes 3 --space image \
      --guidance_scale "${GUIDE_SCALE}" --guidance_start_step 1 \
      --guidance_mode safe_minus_harm --safe_classes 1 --harm_classes 2 \
      --spatial_guidance --spatial_threshold "${thresh}"

    echo "  [img] done."
    echo ""
done

echo "=============================================="
echo "All 3-class sexual spatial inference complete! (thresholds: ${THRESHOLDS})"
echo "=============================================="
