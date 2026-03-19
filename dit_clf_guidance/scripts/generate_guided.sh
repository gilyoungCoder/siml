#!/usr/bin/env bash
# Guided generation for country body prompts at multiple scales (Pony V7)
#
# Usage:
#   bash scripts/generate_guided.sh
#   CLASSIFIER_CKPT=work_dirs/.../classifier.pth bash scripts/generate_guided.sh

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.."

PRETRAINED_MODEL="purplesmartai/pony-v7-base"

# Set classifier checkpoint (override via env variable)
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-work_dirs/pony_z0_resnet18/checkpoint/step_1000/classifier.pth}"

if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "ERROR: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    echo "Train first with: bash scripts/train.sh"
    echo "Or set CLASSIFIER_CKPT env variable."
    exit 1
fi

echo "Using classifier: $CLASSIFIER_CKPT"

for PROMPT_FILE in prompts/country_body.txt prompts/country_nude_body.txt; do
    BASENAME=$(basename "$PROMPT_FILE" .txt)

    for SCALE in 5 10 15 20; do
        OUTPUT_DIR="output_img/${BASENAME}_guided_s${SCALE}"
        echo ""
        echo "=== ${BASENAME}, scale=${SCALE} ==="

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
          --classifier_ckpt "$CLASSIFIER_CKPT" \
          --num_classes 3 \
          --guidance_scale "$SCALE" \
          --guidance_mode safe_minus_harm \
          --safe_classes 0 1 \
          --harm_classes 2 \
          --grad_clip_ratio 0.3
    done
done

echo ""
echo "Done. Guided images saved to output_img/"
