#!/bin/bash

# GenEval with OWL-ViT (Official 553 Prompts)
# mmcv-full 설치 문제로 OWL-ViT 사용

export CUDA_VISIBLE_DEVICES=7

# 공식 GenEval 553 프롬프트 설정
IMG_DIR="/mnt/home/yhgil99/unlearning/vlm/geneval_official_repo/sd1.4_vanilla"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/vlm/geneval_official_repo/prompts/evaluation_metadata.jsonl"
OUTPUT_FILE="geneval_owlvit_sd14_vanilla_results.json"
DEVICE="cuda"

echo "========================================"
echo "GenEval with OWL-ViT (Official 553)"
echo "========================================"
echo "Image Directory: $IMG_DIR"
echo "Prompt Metadata: $PROMPT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo "========================================"
echo ""

/mnt/home/yhgil99/.conda/envs/sdd/bin/python geneval_owlvit.py \
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
