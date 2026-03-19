#!/usr/bin/env bash

# =========================
# UNIFIED Harmful Content Suppression
# Combines: Soft Delete + Latent-Guided + Classifier Guidance
# =========================

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sdd

# Now enable strict mode after conda activation
set -euo pipefail

# GPU Configuration
export CUDA_VISIBLE_DEVICES=5

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/sexual_50.txt"
OUTPUT_DIR="./output_unified/sexual_all_methods_$(date +%Y%m%d_%H%M%S)"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# Shared Classifier
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"

# =========================
# METHOD 1: Soft Delete (Token-based)
# =========================
SOFT_DELETE=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"
ADAPTIVE_THRESHOLD=true
BASE_TAU=0.15
CENTRAL_PERCENTILE=0.80
TAU_FACTOR=1.05
HARM_GAMMA_START=40.0
HARM_GAMMA_END=0.5

# =========================
# METHOD 2: Latent-Guided Suppression
# =========================
LATENT_GUIDED=true
HARMFUL_THRESHOLD=0.0  # Class 2 (nude) logit threshold
ALIGNMENT_THRESHOLD=0.3
SUPPRESSION_STRENGTH=40.0
SUPPRESS_START_STEP=0
SUPPRESS_END_STEP=50

# =========================
# METHOD 3: Classifier Guidance
# =========================
CLASSIFIER_GUIDANCE=false  # Set to true to enable
GUIDANCE_SCALE=7.5
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 0=not-relevant, 1=clothed, 2=nude

# =========================
# DEBUG Options
# =========================
DEBUG=true
DEBUG_STEPS=true
DEBUG_PROMPTS=false

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                 UNIFIED HARMFUL CONTENT SUPPRESSION                            ║"
echo "║     Soft Delete + Latent-Guided + Classifier Guidance (All Methods)           ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo ""
echo "┌─ METHOD 1: Soft Delete (Token-based) ────────────────────────────────────────┐"
echo "│  Enabled: ${SOFT_DELETE}"
if [ "${SOFT_DELETE}" = true ]; then
echo "│  Harmful concepts: ${HARM_CONCEPTS_FILE}"
echo "│  Adaptive threshold: ${ADAPTIVE_THRESHOLD}"
echo "│  Base tau: ${BASE_TAU}"
echo "│  Tau factor: ${TAU_FACTOR}"
echo "│  Gamma schedule: ${HARM_GAMMA_START} → ${HARM_GAMMA_END}"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ METHOD 2: Latent-Guided Suppression ────────────────────────────────────────┐"
echo "│  Enabled: ${LATENT_GUIDED}"
if [ "${LATENT_GUIDED}" = true ]; then
echo "│  Harmful threshold (Class 2 - nude): ${HARMFUL_THRESHOLD}"
echo "│  Alignment threshold: ${ALIGNMENT_THRESHOLD}"
echo "│  Suppression strength: ${SUPPRESSION_STRENGTH}"
echo "│  Steps: ${SUPPRESS_START_STEP} → ${SUPPRESS_END_STEP}"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ METHOD 3: Classifier Guidance ───────────────────────────────────────────────┐"
echo "│  Enabled: ${CLASSIFIER_GUIDANCE}"
if [ "${CLASSIFIER_GUIDANCE}" = true ]; then
echo "│  Scale: ${GUIDANCE_SCALE}"
echo "│  Target class: ${TARGET_CLASS} (0=not-relevant, 1=clothed, 2=nude)"
echo "│  Start step: ${GUIDANCE_START_STEP}"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Build command
ARGS=(
    "${CKPT_PATH}"
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --nsamples "${NSAMPLES}"
    --cfg_scale "${CFG_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
    --classifier_ckpt "${CLASSIFIER_CKPT}"
    --classifier_config "${CLASSIFIER_CONFIG}"
)

# Soft Delete
if [ "${SOFT_DELETE}" = true ]; then
    ARGS+=(
        --soft_delete
        --harm_concepts_file "${HARM_CONCEPTS_FILE}"
        --base_tau "${BASE_TAU}"
        --central_percentile "${CENTRAL_PERCENTILE}"
        --tau_factor "${TAU_FACTOR}"
        --harm_gamma_start "${HARM_GAMMA_START}"
        --harm_gamma_end "${HARM_GAMMA_END}"
    )
    if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
        ARGS+=(--adaptive_threshold)
    fi
fi

# Latent-Guided
if [ "${LATENT_GUIDED}" = true ]; then
    ARGS+=(
        --latent_guided
        --harmful_threshold "${HARMFUL_THRESHOLD}"
        --alignment_threshold "${ALIGNMENT_THRESHOLD}"
        --suppression_strength "${SUPPRESSION_STRENGTH}"
        --suppress_start_step "${SUPPRESS_START_STEP}"
        --suppress_end_step "${SUPPRESS_END_STEP}"
    )
fi

# Classifier Guidance
if [ "${CLASSIFIER_GUIDANCE}" = true ]; then
    ARGS+=(
        --classifier_guidance
        --guidance_scale "${GUIDANCE_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --target_class "${TARGET_CLASS}"
    )
fi

# Debug
if [ "${DEBUG}" = true ]; then
    ARGS+=(--debug)
fi
if [ "${DEBUG_STEPS}" = true ]; then
    ARGS+=(--debug_steps)
fi
if [ "${DEBUG_PROMPTS}" = true ]; then
    ARGS+=(--debug_prompts)
fi

mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_unified_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo "[INFO] Starting unified generation..."
echo "[INFO] Log: ${LOG}"
echo ""

# Run
nohup python generate_unified.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
