#!/usr/bin/env bash
set -euo pipefail

# =========================
# Image Classification Script
# Classify images in a folder using nudity classifier
# =========================

# Check if image directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_directory> [options]"
    echo ""
    echo "Example:"
    echo "  $0 ./output_images"
    echo "  $0 ./output_images --verbose"
    exit 1
fi

IMAGE_DIR="$1"
shift  # Remove first argument

# Default configuration
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
VAE_MODEL="CompVis/stable-diffusion-v1-4"
NUM_CLASSES=3
DEVICE="cuda"
BATCH_SIZE=8

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    IMAGE CLASSIFICATION WITH NUDITY CLASSIFIER                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Image directory: ${IMAGE_DIR}"
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] VAE model: ${VAE_MODEL}"
echo "[CONFIG] Device: ${DEVICE}"
echo ""
echo "┌─ Class Definitions ───────────────────────────────────────────────────────────┐"
echo "│  Class 0: Not-Relevant                                                        │"
echo "│  Class 1: Clothed People                                                      │"
echo "│  Class 2: Nude People                                                         │"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Run classification
python classify_images.py \
    "${IMAGE_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --vae_model "${VAE_MODEL}" \
    --num_classes "${NUM_CLASSES}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --summary \
    "$@"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Classification complete!                                                      ║"
echo "║                                                                                ║"
echo "║  Results saved to:                                                             ║"
echo "║    - ${IMAGE_DIR}/classification_results.json                                  ║"
echo "║    - ${IMAGE_DIR}/classification_results.csv                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
