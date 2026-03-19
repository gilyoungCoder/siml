#!/bin/bash

# GenEval Official 553 Prompts Evaluation
# 공식 GenEval 553개 프롬프트로 생성된 이미지 평가

export CUDA_VISIBLE_DEVICES=7

# 공식 GenEval 553 프롬프트 설정
IMG_DIR="/mnt/home/yhgil99/unlearning/vlm/geneval_official_repo/sd1.4_vanilla"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/vlm/geneval_official_repo/prompts/evaluation_metadata.jsonl"
OUTPUT_FILE="geneval_sd14_vanilla_results.json"
MODEL_PATH="/mnt/home/yhgil99/unlearning/vlm/geneval_models"

echo "========================================"
echo "GenEval Official 553 Prompts Evaluation"
echo "========================================"
echo "Image Directory: $IMG_DIR"
echo "Prompt Metadata: $PROMPT_FILE"
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
