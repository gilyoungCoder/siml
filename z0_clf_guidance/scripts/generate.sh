#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="../prompts/country_nude_body.txt"
OUTPUT_DIR="./output_img/z0_guided"

# Update this to your trained checkpoint
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
  --guidance_scale 5.0 \
  --guidance_start_step 1 \
  --target_class 1
