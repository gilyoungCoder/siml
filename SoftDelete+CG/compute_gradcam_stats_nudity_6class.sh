#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Nudity 6-Class Classifier
#
# Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes)
#   2: harm_full_nude (completely naked)
#   3: harm_topless (topless with jeans)
#   4: harm_lingerie (underwear/lingerie)
#   5: harm_swimwear (revealing bikini)
#
# This script computes per-class GradCAM statistics for harm classes (2,3,4,5)
# ============================================================================

export CUDA_VISIBLE_DEVICES=6

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Configuration - MODIFY THESE PATHS
CLASSIFIER_CKPT="./work_dirs/nudity_6class/checkpoint/step_28000/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"
NUM_CLASSES=6
NUM_SAMPLES=1000
BATCH_SIZE=8

# Data directories for each harm class
DATA_BASE="/mnt/home/yhgil99/dataset/threeclassImg/nudity_6class"
FULL_NUDE_DIR="${DATA_BASE}/completely_naked"
TOPLESS_DIR="${DATA_BASE}/topless_with_jeans"
LINGERIE_DIR="${DATA_BASE}/lingerie"
SWIMWEAR_DIR="${DATA_BASE}/revealing_bikini"

# Output directory
OUTPUT_DIR="./gradcam_stats/nudity_6class"
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Computing GradCAM Statistics for Nudity 6-Class"
echo "=============================================="
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="

# Class 2: harm_full_nude
echo ""
echo "[1/4] Computing stats for class 2 (harm_full_nude)..."
python compute_gradcam_statistics.py \
    --data_dir "${FULL_NUDE_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_full_nude_class2.json" \
    --harmful_class 2 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 3: harm_topless
echo ""
echo "[2/4] Computing stats for class 3 (harm_topless)..."
python compute_gradcam_statistics.py \
    --data_dir "${TOPLESS_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_topless_class3.json" \
    --harmful_class 3 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 4: harm_lingerie
echo ""
echo "[3/4] Computing stats for class 4 (harm_lingerie)..."
python compute_gradcam_statistics.py \
    --data_dir "${LINGERIE_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_lingerie_class4.json" \
    --harmful_class 4 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 5: harm_swimwear
echo ""
echo "[4/4] Computing stats for class 5 (harm_swimwear)..."
python compute_gradcam_statistics.py \
    --data_dir "${SWIMWEAR_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_swimwear_class5.json" \
    --harmful_class 5 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

echo ""
echo "=============================================="
echo "DONE! All GradCAM statistics saved to: ${OUTPUT_DIR}"
echo "=============================================="
echo "Files created:"
ls -la ${OUTPUT_DIR}/
