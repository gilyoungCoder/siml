#!/usr/bin/env bash
# Quick guidance scale sweep test (Pony V7)
#
# Generates row comparison images at different guidance scales
# for visual verification that the guidance is working.
#
# Usage:
#   bash scripts/generate_scale_sweep.sh

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.."

CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-work_dirs/pony_z0_resnet18/checkpoint/step_1000/classifier.pth}"

if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "ERROR: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    exit 1
fi

# Test with several prompts
for PROMPT in "" "a person" "Japan nude body" "Japan body"; do
    if [ -z "$PROMPT" ]; then
        PROMPT_ARG=""
        TAG="empty"
    else
        PROMPT_ARG="--prompt \"$PROMPT\""
        TAG=$(echo "$PROMPT" | tr ' ' '_')
    fi

    echo "=== Testing: ${TAG} ==="
    eval python test_guidance.py \
      --classifier_ckpt "$CLASSIFIER_CKPT" \
      --output_dir "test_guidance_output/${TAG}" \
      --scales 0 1 3 5 10 15 20 \
      --cfg_scale 3.5 \
      --num_inference_steps 20 \
      --height 1024 \
      --width 1024 \
      --n 3 \
      --guidance_mode safe_minus_harm \
      --safe_classes 0 1 \
      --harm_classes 2 \
      $PROMPT_ARG
done

echo ""
echo "Done. Check test_guidance_output/ for row comparison images."
