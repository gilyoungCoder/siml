#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for 4-Class Nudity Classifier
#
# This script computes GradCAM statistics for:
#   - Class 2: harm_nude (nudity images)
#   - Class 3: harm_color (color artifact images)
#
# These statistics are required for proper spatial thresholding during inference.
# Run this BEFORE running generate_nudity_4class_spatial_cg.sh
#
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Classifier checkpoint (step 17100)
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"

# SD model path
SD_MODEL="CompVis/stable-diffusion-v1-4"

# Output directory for statistics
OUTPUT_DIR="./gradcam_stats/nudity_4class"

# Number of samples to use for computing statistics
NUM_SAMPLES=1000

# ============================================================================
# Compute statistics for Class 2 (harm_nude)
# ============================================================================
echo "=============================================="
echo "Computing GradCAM statistics for Class 2 (harm_nude)"
echo "=============================================="

# Use nudity images for class 2
NUDE_DATA_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"

python compute_gradcam_statistics_4class.py \
    --data_dir "${NUDE_DATA_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_class 2 \
    --num_samples ${NUM_SAMPLES} \
    --batch_size 8 \
    --seed 42

# ============================================================================
# Compute statistics for Class 3 (harm_color)
# ============================================================================
echo ""
echo "=============================================="
echo "Computing GradCAM statistics for Class 3 (harm_color)"
echo "=============================================="

# Use color artifact images for class 3
COLOR_DATA_DIR="/mnt/home/yhgil99/dataset/threeclassImg/nudity/color_artifacts_strong"

python compute_gradcam_statistics_4class.py \
    --data_dir "${COLOR_DATA_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --target_class 3 \
    --num_samples ${NUM_SAMPLES} \
    --batch_size 8 \
    --seed 42

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "GradCAM Statistics Computation Complete!"
echo "=============================================="
echo "Output files saved to: ${OUTPUT_DIR}"
echo "  - gradcam_stats_harm_nude_class2.json"
echo "  - gradcam_stats_harm_color_class3.json"
echo ""
echo "Now you can run inference with:"
echo "  ./generate_nudity_4class_spatial_cg.sh"
echo ""
echo "Make sure to add --gradcam_stats_dir ${OUTPUT_DIR} to the inference script!"
echo "=============================================="
