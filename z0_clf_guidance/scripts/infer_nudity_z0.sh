#!/usr/bin/env bash
# ============================================================================
# [INFERENCE] Latent z0-space classifier guidance — nudity (ring-a-bell)
# Classifier: ResNet18 on z0_hat (4ch, 64x64)
# Guidance:   safe_minus_harm  =>  max log p(safe) - log p(harm)
#   safe = [1] (clothed),  harm = [2] (nude)
# ============================================================================
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
OUTPUT_DIR="./output_img/nudity_ring_a_bell_z0_safe_minus_harm"

# Best latent-space 3-class classifier (acc=0.915)
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"

python generate.py "${CKPT_PATH}" \
  --prompt_file "${PROMPT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --nsamples 1 \
  --cfg_scale 7.5 \
  --num_inference_steps 50 \
  --seed 1234 \
  --classifier_ckpt "${CLASSIFIER_CKPT}" \
  --architecture resnet18 \
  --num_classes 3 \
  --space latent \
  --guidance_scale 5.0 \
  --guidance_start_step 1 \
  --guidance_mode safe_minus_harm \
  --safe_classes 1 \
  --harm_classes 2
