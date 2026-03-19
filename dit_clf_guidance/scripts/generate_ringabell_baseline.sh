#!/bin/bash
# Generate baseline images (no guidance) using Ring-a-Bell nudity prompts (Pony V7).
# 80 sensitive prompts from nudity-ring-a-bell.csv, 1 sample each.

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.." || exit 1

python generate.py \
    --pretrained_model_name_or_path "purplesmartai/pony-v7-base" \
    --prompt_csv "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv" \
    --csv_column "sensitive prompt" \
    --output_dir output_img/ringabell_nudity_baseline \
    --nsamples 1 \
    --cfg_scale 3.5 \
    --num_inference_steps 20 \
    --height 1024 \
    --width 1024 \
    --seed 1234 \
    --guidance_scale 0 \
    --mixed_precision bf16
