#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Nudity 6-Class v2 Classifier
#
# v2 Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes)
#   2: harm_nude (completely naked + topless merged)
#   3: harm_lingerie (underwear/lingerie)
#   4: harm_swimwear (revealing bikini)
#   5: harm_color (color artifacts) - NEW
#
# This script computes per-class GradCAM statistics for harm classes (2,3,4,5)
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Configuration
STEP=22700
CLASSIFIER_CKPT="./work_dirs/nudity_6class_v2/checkpoint/step_${STEP}/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"
NUM_CLASSES=6
NUM_SAMPLES=1000
BATCH_SIZE=8

# Data directories for each harm class
DATA_BASE="/mnt/home/yhgil99/dataset/threeclassImg/nudity_6class"
# harm_nude: completely_naked + topless_with_jeans merged -> use completely_naked as representative
NUDE_DIR="${DATA_BASE}/completely_naked"
LINGERIE_DIR="${DATA_BASE}/lingerie"
SWIMWEAR_DIR="${DATA_BASE}/revealing_bikini"
COLOR_ARTIFACTS_DIR="/mnt/home/yhgil99/dataset/threeclassImg/nudity/color_artifacts_strong"

# Output directory
OUTPUT_DIR="./gradcam_stats/nudity_6class_v2_step${STEP}"
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "Computing GradCAM Statistics for Nudity 6-Class v2"
echo "=============================================="
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="

# Check classifier exists
if [ ! -f "${CLASSIFIER_CKPT}" ]; then
    echo "ERROR: Classifier not found: ${CLASSIFIER_CKPT}"
    exit 1
fi

# Class 2: harm_nude (completely naked + topless merged)
echo ""
echo "[1/4] Computing stats for class 2 (harm_nude)..."
python compute_gradcam_statistics.py \
    --data_dir "${NUDE_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_nude_class2.json" \
    --harmful_class 2 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 3: harm_lingerie
echo ""
echo "[2/4] Computing stats for class 3 (harm_lingerie)..."
python compute_gradcam_statistics.py \
    --data_dir "${LINGERIE_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_lingerie_class3.json" \
    --harmful_class 3 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 4: harm_swimwear
echo ""
echo "[3/4] Computing stats for class 4 (harm_swimwear)..."
python compute_gradcam_statistics.py \
    --data_dir "${SWIMWEAR_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_swimwear_class4.json" \
    --harmful_class 4 \
    --num_classes ${NUM_CLASSES} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE}

# Class 5: harm_color (color artifacts)
echo ""
echo "[4/4] Computing stats for class 5 (harm_color)..."
python compute_gradcam_statistics.py \
    --data_dir "${COLOR_ARTIFACTS_DIR}" \
    --pretrained_model_name_or_path "${SD_MODEL}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --output_file "${OUTPUT_DIR}/gradcam_stats_color_class5.json" \
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
