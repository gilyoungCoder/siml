#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier Interpretation - Single Image Mode (VIOLENCE)
# =========================
# 단일 이미지를 분석하여 violence classifier가 어디를 보는지 시각화

# GPU Configuration
export CUDA_VISIBLE_DEVICES=7

# =========================
# Input Configuration
# =========================
# 분석할 이미지 경로 (아래 중 하나를 선택하거나 직접 경로 지정)
IMAGE_PATH="./outputs/violence/vanilla/prompt_0001_sample_1.png"
# IMAGE_PATH="./img/violence_samples/000000.png"
# IMAGE_PATH="./geo_utils/data/test_violence.png"

# =========================
# Model Configuration
# =========================
CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
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
                         # 1 = peaceful people (safe)
                         # 2 = violent people (harmful)

# =========================
# Output Configuration
# =========================
OUTPUT_DIR="./interpretation_results/violence_single_$(basename "${IMAGE_PATH}" .png)"

# =========================
# Run Analysis
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/interpret_violence_single_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║         VIOLENCE CLASSIFIER INTERPRETATION - SINGLE IMAGE MODE                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Image: ${IMAGE_PATH}"
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] Timestep: ${TIMESTEP}"
echo "[CONFIG] Target Class: ${TARGET_CLASS} (0=not people, 1=peaceful, 2=violent)"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""

# Check if image exists
if [ ! -f "${IMAGE_PATH}" ]; then
    echo "❌ Error: Image file not found: ${IMAGE_PATH}"
    echo ""
    echo "Please set IMAGE_PATH to a valid image file."
    echo ""
    echo "Example paths:"
    echo "  - Generated images: ./outputs/violence/*/prompt_*.png"
    echo "  - Custom images: ./img/violence_samples/*.png"
    exit 1
fi

echo "┌─ Analysis Details ────────────────────────────────────────────────────────────┐"
echo "│  1. Grad-CAM: Heatmap showing which latent regions classifier focuses on"
echo "│  2. Layer-wise: Activation statistics at different U-Net layers"
echo "│  3. Integrated Gradients: Attribution per latent channel"
echo "│  4. Summary: Numerical results in JSON format"
echo "│"
echo "│  What will be visualized:"
echo "│    - RED regions: Areas classifier identifies as violent"
echo "│    - Overlay: Grad-CAM heatmap on original image"
echo "│    - Stats: How confident the classifier is at each layer"
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
echo "║  View results:"
echo "║    eog ${OUTPUT_DIR}/*.png"
echo "║"
echo "║  View summary:"
echo "║    cat ${OUTPUT_DIR}/*_summary.json | jq"
echo "║"
echo "║  Interpretation guide:"
echo "║    - Bright red areas = Violence classifier focuses here"
echo "║    - Higher confidence = Classifier is more certain about violence"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
