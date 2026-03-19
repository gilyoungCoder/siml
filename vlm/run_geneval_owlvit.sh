#!/bin/bash

# GenEval with OWL-ViT
# 오리지널 GenEval 평가 로직 + OWL-ViT 객체 탐지

export CUDA_VISIBLE_DEVICES=7

# 기본 설정
IMG_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/outputs/classifier_masked_adversarial_49+CG_percentile0.5+noCG"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/prompts/sexual_50.txt"
OUTPUT_FILE="geneval_owlvit_results.json"
DEVICE="cuda"

# 오리지널 GenEval 파라미터
THRESHOLD=0.3
COUNTING_THRESHOLD=0.9
MAX_OBJECTS=16
NMS_THRESHOLD=1.0
POSITION_THRESHOLD=0.1

echo "========================================"
echo "GenEval with OWL-ViT"
echo "========================================"
echo "Image Directory: $IMG_DIR"
echo "Prompt File: $PROMPT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo ""
echo "Parameters:"
echo "  Threshold: $THRESHOLD"
echo "  Counting Threshold: $COUNTING_THRESHOLD"
echo "  Max Objects: $MAX_OBJECTS"
echo "  NMS Threshold: $NMS_THRESHOLD"
echo "  Position Threshold: $POSITION_THRESHOLD"
echo "========================================"
echo ""

python geneval_owlvit.py \
    --img_dir "$IMG_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output "$OUTPUT_FILE" \
    --device "$DEVICE" \
    --threshold $THRESHOLD \
    --counting_threshold $COUNTING_THRESHOLD \
    --max_objects $MAX_OBJECTS \
    --nms_threshold $NMS_THRESHOLD \
    --position_threshold $POSITION_THRESHOLD

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
