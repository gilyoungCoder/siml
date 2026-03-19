#!/usr/bin/env bash
set -euo pipefail

# =========================
# Machine Unlearning with DEBUG LOGGING
# Detailed token-level analysis and step-by-step tracking
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=4

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/country_nude_body.txt"
OUTPUT_DIR="./output_img/unlearning_debug20, 5_7.5"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Harmful Concept Suppression (Attention Manipulation)
# =========================
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"  # One concept per line (e.g., nude, naked, nudity)
HARM_TAU=0.15                                      # Cosine similarity threshold
HARM_GAMMA_START=20.0                              # Suppression strength at early steps
HARM_GAMMA_END=0.5                                 # Suppression strength at late steps

# =========================
# Classifier Guidance
# =========================
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
GUIDANCE_SCALE=5
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = clothed people

# =========================
# DEBUG Options
# =========================
DEBUG=true              # Enable general debug mode
DEBUG_PROMPTS=true      # Show detailed per-token analysis for each prompt
DEBUG_STEPS=true        # Show per-step suppression statistics during generation

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_debug_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    MACHINE UNLEARNING - DEBUG MODE                             ║"
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
echo "│  Threshold (τ): ${HARM_TAU}"
echo "│  Gamma schedule: ${HARM_GAMMA_START} → ${HARM_GAMMA_END}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Classifier Guidance ─────────────────────────────────────────────────────────┐"
echo "│  Enabled: ${CLASSIFIER_GUIDANCE}"
echo "│  Checkpoint: ${CLASSIFIER_CKPT}"
echo "│  Scale: ${GUIDANCE_SCALE}"
echo "│  Target class: ${TARGET_CLASS} (clothed people)"
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
        --harm_tau "${HARM_TAU}"
        --harm_gamma_start "${HARM_GAMMA_START}"
        --harm_gamma_end "${HARM_GAMMA_END}"
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

echo "[INFO] Starting generation in DEBUG mode..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_debug.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
