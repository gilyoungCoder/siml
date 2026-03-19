#!/bin/bash

# GenEval Score Evaluation Script
# 이미지 디렉토리와 프롬프트 파일을 지정하여 평가를 실행합니다.

# 기본 설정
IMG_DIR=""
PROMPT_FILE=""
OUTPUT_FILE="geneval_results.json"
DEVICE="cuda"

# 사용법 출력
usage() {
    echo "Usage: $0 --img_dir <path> --prompt_file <path> [--output <path>] [--device <cuda|cpu>]"
    echo ""
    echo "Required arguments:"
    echo "  --img_dir       Path to directory containing generated images"
    echo "  --prompt_file   Path to file containing prompts (txt or json)"
    echo ""
    echo "Optional arguments:"
    echo "  --output        Path to output JSON file (default: geneval_results.json)"
    echo "  --device        Device to use: cuda or cpu (default: cuda)"
    echo ""
    echo "Example:"
    echo "  $0 --img_dir ./images --prompt_file ./prompts.txt --output ./results.json"
    exit 1
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --img_dir)
            IMG_DIR="$2"
            shift 2
            ;;
        --prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# 필수 인자 확인
if [ -z "$IMG_DIR" ] || [ -z "$PROMPT_FILE" ]; then
    echo "Error: --img_dir and --prompt_file are required"
    usage
fi

# 경로 존재 확인
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory does not exist: $IMG_DIR"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file does not exist: $PROMPT_FILE"
    exit 1
fi

# 실행
echo "========================================"
echo "GenEval Score Evaluation"
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
