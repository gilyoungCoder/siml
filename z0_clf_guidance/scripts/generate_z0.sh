#!/usr/bin/env bash
# [Z0-LATENT] Generate with latent-space z0 classifier guidance
# Chain rule: zt -> Tweedie -> z0_hat -> Classifier(4ch) -> log_prob
# grad = d(log_prob)/d(z0_hat) * 1/sqrt(alpha_bar)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="../prompts/country_nude_body.txt"
OUTPUT_DIR="./output_img/z0_latent_guided"

# Update to your trained latent-space classifier checkpoint
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_XXXX/classifier.pth"

python generate.py "${CKPT_PATH}" \
  --prompt_file "${PROMPT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --nsamples 1 \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --seed 1234 \
  --classifier_ckpt "${CLASSIFIER_CKPT}" \
  --architecture resnet18 \
  --num_classes 3 \
  --space latent \
  --guidance_scale 5.0 \
  --guidance_start_step 1 \
  --target_class 1
