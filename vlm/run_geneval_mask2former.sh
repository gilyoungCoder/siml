#!/bin/bash

# GenEval with Original Mask2Former
# 공식 GenEval evaluate_images.py를 사용한 평가

export CUDA_VISIBLE_DEVICES=7

# 기본 설정
IMG_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/outputs/classifier_masked_adversarial_49+CG_percentile0.5+noCG"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/prompts/sexual_50.txt"
OUTPUT_FILE="geneval_mask2former_results.json"
MODEL_PATH="/mnt/home/yhgil99/unlearning/vlm/geneval_models"

echo "========================================"
echo "GenEval with Original Mask2Former"
echo "========================================"
echo "Image Directory: $IMG_DIR"
echo "Prompt File: $PROMPT_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Model Path: $MODEL_PATH"
echo "========================================"
echo ""

/mnt/home/yhgil99/.conda/envs/sdd/bin/python geneval_mask2former.py \
    --img_dir "$IMG_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --output "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH"

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
