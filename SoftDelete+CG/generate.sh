#!/usr/bin/env bash
set -euo pipefail

# =========================
# Machine Unlearning via Attention Manipulation + Classifier Guidance
# Clean, minimal implementation
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=3

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/country_nude_body.txt"
OUTPUT_DIR="./output_img/unlearning_clean"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Harmful Concept Suppression (Attention Manipulation)
# =========================
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"  # One concept per line (e.g., nude, naked, nudity)
HARM_TAU=0.15                                      # Cosine similarity threshold
HARM_GAMMA_START=40.0                              # Suppression strength at early steps
HARM_GAMMA_END=0.5                                 # Suppression strength at late steps

# =========================
# Classifier Guidance
# =========================
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
GUIDANCE_SCALE=5.0
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = clothed people

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo "[INFO] Starting generation..."
echo "  Model: ${CKPT_PATH}"
echo "  Prompts: ${PROMPT_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG}"
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
    echo "[INFO] Harmful concept suppression: ENABLED"
    echo "  Concepts file: ${HARM_CONCEPTS_FILE}"
    echo "  Tau: ${HARM_TAU}"
    echo "  Gamma: ${HARM_GAMMA_START} -> ${HARM_GAMMA_END}"
else
    echo "[INFO] Harmful concept suppression: DISABLED"
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
    echo "[INFO] Classifier guidance: ENABLED"
    echo "  Checkpoint: ${CLASSIFIER_CKPT}"
    echo "  Scale: ${GUIDANCE_SCALE}"
    echo "  Target class: ${TARGET_CLASS}"
else
    echo "[INFO] Classifier guidance: DISABLED"
fi

echo ""
echo "[INFO] Running generation script..."

# Run in background and save log
nohup python generate.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "[INFO] Process started with PID: ${PID}"
echo "[INFO] Monitor progress with: tail -f \"${LOG}\""
echo ""
echo "To stop the process, run: kill ${PID}"
