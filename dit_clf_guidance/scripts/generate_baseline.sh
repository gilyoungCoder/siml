#!/usr/bin/env bash
# Baseline: generate country body images WITHOUT classifier guidance (Pony V7)
#
# Usage:
#   bash scripts/generate_baseline.sh

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.."

PRETRAINED_MODEL="purplesmartai/pony-v7-base"

for PROMPT_FILE in prompts/country_body.txt prompts/country_nude_body.txt; do
    BASENAME=$(basename "$PROMPT_FILE" .txt)
    OUTPUT_DIR="output_img/${BASENAME}_baseline"

    echo "=== Baseline: ${BASENAME} ==="
    python generate.py \
      --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
      --prompt_file "$PROMPT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --nsamples 1 \
      --cfg_scale 3.5 \
      --num_inference_steps 20 \
      --height 1024 \
      --width 1024 \
      --seed 1234 \
      --guidance_scale 0
done

echo "Done. Baseline images saved."
