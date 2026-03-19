#!/usr/bin/env bash
set -euo pipefail

# =========================
# Selective Classifier Guidance - BENIGN PROMPTS TEST
#
# Purpose: Test on benign prompts to verify minimal intervention
# Expected: High GENEVAL scores (no degradation)
# =========================

export CUDA_VISIBLE_DEVICES=6

# =========================
# Model & I/O
# =========================
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/benign.txt"  # ⚠ UPDATE THIS PATH
OUTPUT_DIR="./outputs/selective_cg_benign_test"

# =========================
# Generation Parameters
# =========================
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Selective Guidance - Conservative Settings for Benign
# =========================
SELECTIVE_GUIDANCE=true
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# Higher threshold = less likely to trigger (good for benign)
HARMFUL_THRESHOLD=0.7  # Conservative (was 0.5)

HARMFUL_CLASS=2
SAFE_CLASS=1

# Spatial masking
USE_PERCENTILE=true
SPATIAL_PERCENTILE=0.2  # Only top 20% (was 0.3)

# Lighter guidance
GUIDANCE_SCALE=5.0  # Reduced (was 5.0)
GUIDANCE_START_STEP=5   # Start later
GUIDANCE_END_STEP=45    # End earlier

GRADCAM_LAYER="encoder_model.middle_block.2"

# =========================
# Debug
# =========================
DEBUG=true
SAVE_VISUALIZATIONS=true

# =========================
# Run
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/benign_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║               BENIGN PROMPTS TEST - SELECTIVE GUIDANCE                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[TEST] Evaluating selective guidance on BENIGN prompts"
echo "[GOAL] Minimal intervention → High GENEVAL scores"
echo ""
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Conservative settings:"
echo "  - Harmful threshold: ${HARMFUL_THRESHOLD} (higher = less intervention)"
echo "  - Guidance scale: ${GUIDANCE_SCALE} (lower = lighter guidance)"
echo "  - Active steps: ${GUIDANCE_START_STEP}-${GUIDANCE_END_STEP} (narrower window)"
echo ""

ARGS=(
    "${CKPT_PATH}"
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --nsamples "${NSAMPLES}"
    --cfg_scale "${CFG_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
)

if [ "${SELECTIVE_GUIDANCE}" = true ]; then
    ARGS+=(
        --selective_guidance
        --classifier_ckpt "${CLASSIFIER_CKPT}"
        --harmful_threshold "${HARMFUL_THRESHOLD}"
        --harmful_class "${HARMFUL_CLASS}"
        --safe_class "${SAFE_CLASS}"
        --spatial_threshold 0.5
        --use_percentile
        --spatial_percentile "${SPATIAL_PERCENTILE}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --guidance_end_step "${GUIDANCE_END_STEP}"
        --gradcam_layer "${GRADCAM_LAYER}"
    )
fi

if [ "${DEBUG}" = true ]; then
    ARGS+=(--debug)
fi

if [ "${SAVE_VISUALIZATIONS}" = true ]; then
    ARGS+=(--save_visualizations)
fi

echo "[INFO] Starting benign test..."
nohup python generate_selective_cg.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Benign test started with PID: ${PID}"
echo "║"
echo "║  Monitor:  tail -f \"${LOG}\""
echo "║  Output:   ${OUTPUT_DIR}"
echo "║"
echo "║  Next steps after generation:"
echo "║    1. Check log for guidance statistics"
echo "║       → Expected: Low guidance_ratio (most steps skipped)"
echo "║    2. Run GENEVAL evaluation"
echo "║       → Expected: High scores (minimal degradation)"
echo "║    3. Visually inspect images"
echo "║       → Expected: Clean, unaffected generation"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
