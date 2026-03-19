#!/usr/bin/env bash
# ============================================================================
# [INFERENCE] 3-class nudity clf on sexual prompts — guidance_scale sweep
# Classifier: ResNet18 (3-class: 0=non-people, 1=clothed, 2=nude)
# Guidance:   safe_minus_harm  =>  safe=[1], harm=[2]
# Scales:     10, 12.5, 15
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

SCALES="10.0 12.5 15.0"

for scale in ${SCALES}; do
    tag=$(echo "${scale}" | sed 's/\.0$//' | sed 's/\./_/')

    echo "=============================================="
    echo "[3-class sexual] guidance_scale=${scale}  (tag=s${tag})"
    echo "=============================================="

    # --- latent z0 ---
    echo "  [z0] starting..."
    python generate.py "${CKPT_PATH}" \
      --prompt_file "${PROMPT_FILE}" \
      --output_dir "./output_img/sexual_z0_3class_s${tag}" \
      --nsamples 1 --cfg_scale 7.5 --num_inference_steps 50 --seed 1234 \
      --classifier_ckpt "${Z0_CLF}" \
      --architecture resnet18 --num_classes 3 --space latent \
      --guidance_scale "${scale}" --guidance_start_step 1 \
      --guidance_mode safe_minus_harm --safe_classes 1 --harm_classes 2

    echo "  [z0] done."

    # --- image space ---
    echo "  [img] starting..."
    python generate.py "${CKPT_PATH}" \
      --prompt_file "${PROMPT_FILE}" \
      --output_dir "./output_img/sexual_img_3class_s${tag}" \
      --nsamples 1 --cfg_scale 7.5 --num_inference_steps 50 --seed 1234 \
      --classifier_ckpt "${IMG_CLF}" \
      --architecture resnet18 --num_classes 3 --space image \
      --guidance_scale "${scale}" --guidance_start_step 1 \
      --guidance_mode safe_minus_harm --safe_classes 1 --harm_classes 2

    echo "  [img] done."
    echo ""
done

echo "=============================================="
echo "All 3-class sexual inference complete! (scales: ${SCALES})"
echo "=============================================="
