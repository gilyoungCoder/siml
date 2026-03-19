#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier Interpretation - Batch Mode
# =========================
# 디렉토리의 여러 이미지를 배치로 분석

# GPU Configuration
export CUDA_VISIBLE_DEVICES=7

# =========================
# Input Configuration
# =========================
# 분석할 이미지 디렉토리 (아래 중 하나를 선택하세요)
IMAGE_DIR="./img/CNBWON"
# IMAGE_DIR="./img/country nude body, fully clothed"
# IMAGE_DIR="./SDCGlegacy/i2psexual_CG_previous_15"

# =========================
# Model Configuration
# =========================
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# =========================
# Analysis Parameters
# =========================
TIMESTEP=0                # 분석할 timestep (0-999, 0 = clean image without noise)
TARGET_CLASS=2           # 2 = nude people

# =========================
# Output Configuration
# =========================
OUTPUT_DIR="./interpretation_results/batch_$(basename "${IMAGE_DIR}")"

# =========================
# Run Analysis
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/interpret_batch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║              CLASSIFIER INTERPRETATION - BATCH MODE                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Image Directory: ${IMAGE_DIR}"
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] Timestep: ${TIMESTEP}"
echo "[CONFIG] Target Class: ${TARGET_CLASS} (nude)"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""

# Check if directory exists
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "❌ Error: Image directory not found: ${IMAGE_DIR}"
    echo ""
    echo "Available directories:"
    find ./img -type d -maxdepth 1
    exit 1
fi

# Count images
NUM_PNG=$(find "${IMAGE_DIR}" -maxdepth 1 -name "*.png" | wc -l)
NUM_JPG=$(find "${IMAGE_DIR}" -maxdepth 1 -name "*.jpg" | wc -l)
TOTAL_IMAGES=$((NUM_PNG + NUM_JPG))

if [ "${TOTAL_IMAGES}" -eq 0 ]; then
    echo "❌ Error: No images found in ${IMAGE_DIR}"
    exit 1
fi

echo "┌─ Batch Info ──────────────────────────────────────────────────────────────────┐"
echo "│  Total images: ${TOTAL_IMAGES} (${NUM_PNG} PNG, ${NUM_JPG} JPG)"
echo "│  Estimated time: ~$((TOTAL_IMAGES * 30 / 60)) minutes"
echo "│"
echo "│  Analysis per image:"
echo "│    - Grad-CAM heatmap"
echo "│    - Layer-wise activations"
echo "│    - Integrated Gradients"
echo "│    - Summary JSON"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "[INFO] Starting batch analysis..."
echo "[INFO] Progress will be shown below:"
echo ""

# Run interpretation
python interpret_classifier.py \
    --mode image \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --sd_model "${SD_MODEL}" \
    --timestep "${TIMESTEP}" \
    --target_class "${TARGET_CLASS}" \
    --device cuda \
    2>&1 | tee "${LOG}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Batch analysis complete!"
echo "║"
echo "║  Results saved to: ${OUTPUT_DIR}"
echo "║"
echo "║  Each image has its own subdirectory with:"
echo "║    - *_gradcam.png"
echo "║    - *_layers.png"
echo "║    - *_integrated_gradients.png"
echo "║    - *_summary.json"
echo "║"
echo "║  Quick stats:"
echo "║"

# Generate statistics
python << 'EOF'
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
summaries = list(output_dir.glob("*/summary.json"))

if not summaries:
    print("║    No summary files found.")
    sys.exit(0)

nude_probs = []
predicted_classes = []

for summary_path in summaries:
    with open(summary_path) as f:
        data = json.load(f)
        nude_probs.append(data['predictions']['probs'][2])
        predicted_classes.append(data['predictions']['predicted_class'])

print(f"║    Analyzed images: {len(summaries)}")
print(f"║    Average nude probability: {sum(nude_probs)/len(nude_probs):.3f}")
print(f"║    Max nude probability: {max(nude_probs):.3f}")
print(f"║    Min nude probability: {min(nude_probs):.3f}")
print(f"║")
print(f"║    Predicted as nude: {predicted_classes.count(2)}")
print(f"║    Predicted as clothed: {predicted_classes.count(1)}")
print(f"║    Predicted as not people: {predicted_classes.count(0)}")

EOF "${OUTPUT_DIR}"

echo "║"
echo "║  View all results:  find ${OUTPUT_DIR} -name '*_gradcam.png'"
echo "║  Aggregate stats:   find ${OUTPUT_DIR} -name 'summary.json' -exec cat {} \;"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
