#!/usr/bin/env bash
set -euo pipefail

# =========================
# Machine Unlearning with ADAPTIVE THRESHOLD - VIOLENCE
# SAFREE-style dynamic threshold per prompt
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=3

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/violence_50.txt"
OUTPUT_DIR="./SDCG_analysis/violence/SD60_CG_single"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Harmful Concept Suppression (Attention Manipulation)
# =========================
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_violence.txt"

# =========================
# ADAPTIVE THRESHOLD (NEW!)
# =========================
ADAPTIVE_THRESHOLD=true          # Enable adaptive threshold computation
BASE_TAU=0.15                    # Base threshold (fallback)
CENTRAL_PERCENTILE=0.80          # Use central 80% for mean calculation (removes top/bottom 10%)
TAU_FACTOR=1.05                  # Multiplicative factor: tau = central_mean × factor

# If ADAPTIVE_THRESHOLD=false, BASE_TAU will be used as fixed threshold
# If ADAPTIVE_THRESHOLD=true, threshold is computed per prompt as:
#   1. Compute cosine similarities for all content tokens
#   2. Sort similarities and take central 80% (remove outliers)
#   3. Compute mean of central portion
#   4. adaptive_tau = central_mean × TAU_FACTOR (MULTIPLICATIVE!)
#
# Why multiplicative?
#   - Scale-invariant: adapts to similarity distribution magnitude
#   - Proportional: consistent relative increase (e.g., 2% above mean)
#   - Better than additive offset which is fixed absolute value

# Gamma schedule (same as before)
HARM_GAMMA_START=60.0
HARM_GAMMA_END=0.5

# =========================
# Classifier Guidance (3-CLASS VIOLENCE)
# =========================
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
CLASSIFIER_CKPT="./work_dirs/violence_three_class_imgp/checkpoint/step_12400/classifier.pth"
GUIDANCE_SCALE=7.5
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = non-violent/safe content

# =========================
# DEBUG Options
# =========================
DEBUG=true              # Enable general debug mode
DEBUG_PROMPTS=true      # Show detailed per-token analysis with adaptive threshold info
DEBUG_STEPS=true        # Show per-step suppression statistics

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_adaptive_violence_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║         MACHINE UNLEARNING - ADAPTIVE THRESHOLD MODE (VIOLENCE)                ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""
echo "┌─ Harmful Concept Suppression ────────────────────────────────────────────────┐"
echo "│  Enabled: ${HARM_SUPPRESS}"
echo "│  Concepts file: ${HARM_CONCEPTS_FILE}"
if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
    echo "│  "
    echo "│  Threshold Mode: ADAPTIVE (Multiplicative) ⭐"
    echo "│    - Base τ (fallback): ${BASE_TAU}"
    echo "│    - Central percentile: ${CENTRAL_PERCENTILE} (${CENTRAL_PERCENTILE}*100% central data)"
    echo "│    - Multiplicative factor: ×${TAU_FACTOR}"
    echo "│    - Formula: τ_adaptive = central_mean(${CENTRAL_PERCENTILE}) × ${TAU_FACTOR}"
    echo "│    - Scale-invariant, proportional thresholding"
    echo "│    - Per-prompt dynamic threshold"
else
    echo "│  Threshold Mode: FIXED"
    echo "│    - Fixed τ: ${BASE_TAU}"
fi
echo "│  Gamma schedule: ${HARM_GAMMA_START} → ${HARM_GAMMA_END}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Classifier Guidance (3-CLASS VIOLENCE) ──────────────────────────────────────┐"
echo "│  Enabled: ${CLASSIFIER_GUIDANCE}"
echo "│  Checkpoint: ${CLASSIFIER_CKPT}"
echo "│  Scale: ${GUIDANCE_SCALE}"
echo "│  Target class: ${TARGET_CLASS} (non-violent/safe)"
echo "│  Start step: ${GUIDANCE_START_STEP}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Debug Options ───────────────────────────────────────────────────────────────┐"
echo "│  General debug: ${DEBUG}"
echo "│  Per-prompt token analysis: ${DEBUG_PROMPTS}"
echo "│  Per-step statistics: ${DEBUG_STEPS}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Build command arguments
ARGS=(
    "${CKPT_PATH}"
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --nsamples "${NSAMPLES}"
    --cfg_scale "${CFG_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
)

# Add harmful concept suppression arguments
if [ "${HARM_SUPPRESS}" = true ]; then
    ARGS+=(
        --harm_suppress
        --harm_concepts_file "${HARM_CONCEPTS_FILE}"
        --base_tau "${BASE_TAU}"
        --harm_gamma_start "${HARM_GAMMA_START}"
        --harm_gamma_end "${HARM_GAMMA_END}"
    )
fi

# Add adaptive threshold arguments
if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
    ARGS+=(
        --adaptive_threshold
        --central_percentile "${CENTRAL_PERCENTILE}"
        --tau_factor "${TAU_FACTOR}"
    )
fi

# Add classifier guidance arguments
if [ "${CLASSIFIER_GUIDANCE}" = true ]; then
    ARGS+=(
        --classifier_guidance
        --classifier_config "${CLASSIFIER_CONFIG}"
        --classifier_ckpt "${CLASSIFIER_CKPT}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --target_class "${TARGET_CLASS}"
    )
fi

# Add debug flags
if [ "${DEBUG}" = true ]; then
    ARGS+=(--debug)
fi

if [ "${DEBUG_PROMPTS}" = true ]; then
    ARGS+=(--debug_prompts)
fi

if [ "${DEBUG_STEPS}" = true ]; then
    ARGS+=(--debug_steps)
fi

echo "[INFO] Starting generation with ADAPTIVE THRESHOLD mode (VIOLENCE)..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_adaptive.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
