#!/bin/bash
# Generate guided images using Ring-a-Bell nudity prompts (Pony V7).
# Sweeps guidance scales [5, 10, 15, 20].
# Requires a trained classifier checkpoint.

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.." || exit 1

CLASSIFIER_CKPT="${1:?Usage: $0 <classifier_ckpt_path>}"

for SCALE in 5 10 15 20; do
    echo "=== Guidance scale: ${SCALE} ==="
    python generate.py \
        --pretrained_model_name_or_path "purplesmartai/pony-v7-base" \
        --prompt_csv "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv" \
        --csv_column "sensitive prompt" \
        --output_dir "output_img/ringabell_nudity_guided_s${SCALE}" \
        --nsamples 1 \
        --cfg_scale 3.5 \
        --num_inference_steps 20 \
        --height 1024 \
        --width 1024 \
        --seed 1234 \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --num_classes 3 \
        --guidance_scale "$SCALE" \
        --guidance_mode safe_minus_harm \
        --safe_classes 0 1 \
        --harm_classes 2 \
        --grad_clip_ratio 0.3 \
        --mixed_precision bf16
done
