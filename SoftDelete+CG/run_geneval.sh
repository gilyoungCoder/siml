#!/bin/bash

# GenEval Evaluation Script
# =========================
#
# Evaluates generated images using GenEval framework for compositional alignment.
#
# Usage:
#   bash run_geneval.sh
#
# Before running:
#   1. Clone GenEval repository:
#      git clone https://github.com/djghosh13/geneval.git ./geneval
#
#   2. Install dependencies (follow GenEval README):
#      cd geneval
#      conda env create -f environment.yml
#      conda activate geneval
#      # Install mmdetection, mmcv, etc. (see GenEval docs)
#
#   3. Download object detector models:
#      ./evaluation/download_models.sh ./models
#

# Configuration
# =============
export CUDA_VISIBLE_DEVICES=7

# Path to GenEval repository (clone from https://github.com/djghosh13/geneval)
GENEVAL_PATH="./geneval"

# Path to object detector models (downloaded via GenEval's download_models.sh)
DETECTOR_PATH="./geneval/models"

# Directory containing generated images
IMAGE_DIR="./outputs/classifier_masked_adversarial_49+CG_percentile0.5"

# Prompt file used for generation
PROMPT_FILE="./prompts/sexual_50.txt"

# Output directory for GenEval results
OUTPUT_DIR="./geneval_results/classifier_masked_adversarial"

# Image pattern to match
IMAGE_PATTERN="*.png"

# Output summary JSON file
OUTPUT_FILE="${OUTPUT_DIR}/summary_scores.json"


# Run GenEval Evaluation
# ======================

echo "======================================"
echo "GenEval Evaluation"
echo "======================================"
echo ""
echo "Image directory: ${IMAGE_DIR}"
echo "Prompt file: ${PROMPT_FILE}"
echo "GenEval path: ${GENEVAL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if GenEval repository exists
if [ ! -d "${GENEVAL_PATH}" ]; then
    echo "ERROR: GenEval repository not found at ${GENEVAL_PATH}"
    echo ""
    echo "Please clone GenEval first:"
    echo "  git clone https://github.com/djghosh13/geneval.git ${GENEVAL_PATH}"
    echo ""
    echo "Then follow installation instructions:"
    echo "  cd ${GENEVAL_PATH}"
    echo "  conda env create -f environment.yml"
    echo "  conda activate geneval"
    echo "  # Install mmdetection, mmcv, etc."
    echo "  ./evaluation/download_models.sh ./models"
    echo ""
    exit 1
fi

# Check if detector models exist
if [ ! -d "${DETECTOR_PATH}" ]; then
    echo "WARNING: Detector models not found at ${DETECTOR_PATH}"
    echo ""
    echo "Please download models first:"
    echo "  cd ${GENEVAL_PATH}"
    echo "  ./evaluation/download_models.sh ./models"
    echo ""
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
python evaluate_geneval.py \
    --image_dir "${IMAGE_DIR}" \
    --prompt_file "${PROMPT_FILE}" \
    --geneval_path "${GENEVAL_PATH}" \
    --detector_path "${DETECTOR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_pattern "${IMAGE_PATTERN}" \
    --output_file "${OUTPUT_FILE}"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "GenEval evaluation completed!"
    echo "======================================"
    echo ""
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""

    # Display results if jq is available
    if command -v jq &> /dev/null; then
        echo "Summary scores:"
        cat "${OUTPUT_FILE}" | jq .
    fi
else
    echo ""
    echo "======================================"
    echo "ERROR: GenEval evaluation failed!"
    echo "======================================"
    echo ""
    echo "Common issues:"
    echo "  1. GenEval not installed properly"
    echo "  2. Object detector models not downloaded"
    echo "  3. Dependencies missing (mmdet, mmcv, etc.)"
    echo ""
    echo "See GenEval documentation: https://github.com/djghosh13/geneval"
    echo ""
    exit 1
fi
