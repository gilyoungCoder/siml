#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier Interpretation - Single Image Mode
# =========================
# 단일 이미지를 분석하여 classifier가 어디를 보는지 시각화

# GPU Configuration
export CUDA_VISIBLE_DEVICES=4

# =========================
# Input Configuration
# =========================
# 분석할 이미지 경로 (아래 중 하나를 선택하세요)
IMAGE_PATH="./img/CNBWON/000000.png"
# IMAGE_PATH="./img/country nude body, fully clothed/000000.png"
# IMAGE_PATH="./geo_utils/data/test.png"

# =========================
# Model Configuration
# =========================
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# =========================
# Analysis Parameters
# =========================
TIMESTEP=500              # 분석할 timestep (0-999)
                         # 500 = mid-diffusion (추천)
                         # 100 = late denoising (fine detail)
                         # 900 = early denoising (high-level structure)

TARGET_CLASS=2           # 분석 대상 클래스
                         # 0 = not people
                         # 1 = clothed people
                         # 2 = nude people (기본값)

# =========================
# Output Configuration
# =========================
OUTPUT_DIR="./interpretation_results/single_$(basename "${IMAGE_PATH}" .png)"

# =========================
# Run Analysis
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/interpret_single_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║              CLASSIFIER INTERPRETATION - SINGLE IMAGE MODE                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Image: ${IMAGE_PATH}"
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] Timestep: ${TIMESTEP}"
echo "[CONFIG] Target Class: ${TARGET_CLASS} (0=not people, 1=clothed, 2=nude)"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""

# Check if image exists
if [ ! -f "${IMAGE_PATH}" ]; then
    echo "❌ Error: Image file not found: ${IMAGE_PATH}"
    echo ""
    echo "Available images in ./img/CNBWON/:"
    ls -1 ./img/CNBWON/*.png | head -5
    echo "..."
    exit 1
fi

echo "┌─ Analysis Details ────────────────────────────────────────────────────────────┐"
echo "│  1. Grad-CAM: Heatmap showing which latent regions classifier focuses on"
echo "│  2. Layer-wise: Activation statistics at different U-Net layers"
echo "│  3. Integrated Gradients: Attribution per latent channel"
echo "│  4. Summary: Numerical results in JSON format"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "[INFO] Starting analysis..."
echo "[INFO] This will take ~30 seconds..."
echo ""

# Run interpretation
python interpret_classifier.py \
    --mode image \
    --image_path "${IMAGE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --sd_model "${SD_MODEL}" \
    --timestep "${TIMESTEP}" \
    --target_class "${TARGET_CLASS}" \
    --device cuda \
    2>&1 | tee "${LOG}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Analysis complete!"
echo "║"
echo "║  Results saved to: ${OUTPUT_DIR}"
echo "║"
echo "║  Files generated:"
echo "║    - $(basename "${IMAGE_PATH}" .png)_gradcam.png              : Grad-CAM heatmap overlay"
echo "║    - $(basename "${IMAGE_PATH}" .png)_layers.png               : Layer activation stats"
echo "║    - $(basename "${IMAGE_PATH}" .png)_integrated_gradients.png : IG attribution"
echo "║    - $(basename "${IMAGE_PATH}" .png)_summary.json             : Numerical results"
echo "║"
echo "║  View results:  eog ${OUTPUT_DIR}/*.png"
echo "║  View summary:  cat ${OUTPUT_DIR}/*_summary.json | jq"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
