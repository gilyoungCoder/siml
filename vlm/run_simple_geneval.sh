#!/bin/bash

# Simple GenEval (CLIP Score only)
# 원래 잘 작동하던 CLIP Score 기반 평가

export CUDA_VISIBLE_DEVICES=7

# 기본 설정
IMG_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/outputs/classifier_masked_adversarial_49+CG_percentile0.5+noCG"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/prompts/sexual_50.txt"
OUTPUT_FILE="geneval_clip_results.json"
DEVICE="cuda"

echo "========================================"
echo "GenEval (CLIP Score Based)"
echo "========================================"
echo "Image Directory: $IMG_DIR"
echo "Prompt File: $PROMPT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo "========================================"
echo ""

python geneval_score.py \
    --img_dir "$IMG_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output "$OUTPUT_FILE" \
    --device "$DEVICE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Evaluation failed with exit code: $EXIT_CODE"
    echo "========================================"
fi

exit $EXIT_CODE
