#!/usr/bin/env bash
set -euo pipefail

# =========================
# Latent-Guided Harmful Content Suppression
# Real-time latent monitoring + Direct attention suppression
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=5

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/sexual_50.txt"
OUTPUT_DIR="./output_latent_guided/sexual_LDCG"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# LATENT-GUIDED SUPPRESSION
# =========================
LATENT_SUPPRESS=true
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# Detection threshold (Class 2 logit - nude people)
HARMFUL_THRESHOLD=0.0  # Logit > 0 means harmful detected

# Alignment threshold (cosine similarity between latent and attention keys)
ALIGNMENT_THRESHOLD=0.3  # Higher = more selective suppression

# Suppression strength
SUPPRESSION_STRENGTH=100.0  # Higher = harder block (adds negative bias to attention)

# Suppression schedule
SUPPRESS_START_STEP=0
SUPPRESS_END_STEP=50

# =========================
# CLASSIFIER GUIDANCE (Optional, can combine with latent suppression)
# =========================
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
GUIDANCE_SCALE=7.5
GUIDANCE_START_STEP=1
TARGET_CLASS=0  # 0 = safe class

# =========================
# DEBUG Options
# =========================
DEBUG=true
DEBUG_STEPS=true

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_latent_guided_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║              LATENT-GUIDED HARMFUL CONTENT SUPPRESSION                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""
echo "┌─ Latent-Guided Suppression ───────────────────────────────────────────────────┐"
echo "│  Enabled: ${LATENT_SUPPRESS}"
echo "│  Classifier: ${CLASSIFIER_CKPT}"
echo "│  "
echo "│  Detection:"
echo "│    - Harmful threshold (Class 2 logit - nude people): ${HARMFUL_THRESHOLD}"
echo "│    - If logit > threshold → harmful detected"
echo "│  "
echo "│  Suppression:"
echo "│    - Alignment threshold (cosine sim): ${ALIGNMENT_THRESHOLD}"
echo "│    - Suppression strength: ${SUPPRESSION_STRENGTH}"
echo "│    - Active steps: ${SUPPRESS_START_STEP} → ${SUPPRESS_END_STEP}"
echo "│  "
echo "│  Mechanism:"
echo "│    1. Monitor latent with classifier at each step"
echo "│    2. If harmful: compute latent-attention alignment"
echo "│    3. Hard suppress attentions aligned with harmful latent"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Classifier Guidance (Optional) ──────────────────────────────────────────────┐"
echo "│  Enabled: ${CLASSIFIER_GUIDANCE}"
echo "│  Scale: ${GUIDANCE_SCALE}"
echo "│  Target class: ${TARGET_CLASS} (0=clothed, 1=not-relevant, 2=nude)"
echo "│  Start step: ${GUIDANCE_START_STEP}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Generation Parameters ───────────────────────────────────────────────────────┐"
echo "│  Steps: ${NUM_INFERENCE_STEPS}"
echo "│  CFG Scale: ${CFG_SCALE}"
echo "│  Samples per prompt: ${NSAMPLES}"
echo "│  Seed: ${SEED}"
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

# Add latent suppression arguments
if [ "${LATENT_SUPPRESS}" = true ]; then
    ARGS+=(
        --latent_suppress
        --classifier_ckpt "${CLASSIFIER_CKPT}"
        --harmful_threshold "${HARMFUL_THRESHOLD}"
        --alignment_threshold "${ALIGNMENT_THRESHOLD}"
        --suppression_strength "${SUPPRESSION_STRENGTH}"
        --suppress_start_step "${SUPPRESS_START_STEP}"
        --suppress_end_step "${SUPPRESS_END_STEP}"
    )
fi

# Add classifier guidance arguments
if [ "${CLASSIFIER_GUIDANCE}" = true ]; then
    ARGS+=(
        --classifier_guidance
        --classifier_config "${CLASSIFIER_CONFIG}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --target_class "${TARGET_CLASS}"
    )
fi

# Add debug flags
if [ "${DEBUG}" = true ]; then
    ARGS+=(--debug)
fi

if [ "${DEBUG_STEPS}" = true ]; then
    ARGS+=(--debug_steps)
fi

echo "[INFO] Starting generation with LATENT-GUIDED SUPPRESSION..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_latent_guided.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
