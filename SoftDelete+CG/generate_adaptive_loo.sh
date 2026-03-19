#!/usr/bin/env bash
set -euo pipefail

# =========================
# Machine Unlearning with LOO CRITICALITY + ADAPTIVE THRESHOLD
# (Probability Redistribution)
# =========================

export CUDA_VISIBLE_DEVICES=2

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/country_nude_body.txt"
OUTPUT_DIR="./SDCGloo_method/CNB_hd2_PR"   # PR: Prob-Redistribution

# Generation
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
SEED=1234

# Harmful Concept Suppression
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"

# ADAPTIVE THRESHOLD
ADAPTIVE_THRESHOLD=true
BASE_TAU=0.15
CENTRAL_PERCENTILE=0.80
TAU_FACTOR=1.05

# LOO Method
LOO_METHOD="harm_distance"

# Gamma schedule (prob-space; 과도 억제 방지용 완만한 값)
HARM_GAMMA_START=20.0
HARM_GAMMA_END=0.5

# Redistribution options
TOPK_MIN=3                 # 최소 억제 보장(양의 criticality)
REDISTRIBUTE_MODE="proportional"  # or "uniform"

# Classifier Guidance
CLASSIFIER_GUIDANCE=false
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
GUIDANCE_SCALE=7.5
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = clothed people

# DEBUG
DEBUG=true
DEBUG_PROMPTS=true
DEBUG_STEPS=true

# =========================
# Run
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/method/run_adaptive_PR_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║     MACHINE UNLEARNING - ADAPTIVE THRESHOLD (Probability Redistribution)      ║"
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
echo "│  LOO Method: ${LOO_METHOD} ⚡"
if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
    echo "│  Threshold Mode: ADAPTIVE (Multiplicative) ⭐"
    echo "│    - Base τ (fallback): ${BASE_TAU}"
    echo "│    - Central percentile: ${CENTRAL_PERCENTILE}"
    echo "│    - Multiplicative factor: ×${TAU_FACTOR}"
else
    echo "│  Threshold Mode: FIXED (τ=${BASE_TAU})"
fi
echo "│  Gamma schedule (prob-space): ${HARM_GAMMA_START} → ${HARM_GAMMA_END}"
echo "│  Redistribution: mode=${REDISTRIBUTE_MODE}, topk_min=${TOPK_MIN}"
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

ARGS=(
  "${CKPT_PATH}"
  --prompt_file "${PROMPT_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --nsamples "${NSAMPLES}"
  --cfg_scale "${CFG_SCALE}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
)

if [ "${HARM_SUPPRESS}" = true ]; then
  ARGS+=(
    --harm_suppress
    --harm_concepts_file "${HARM_CONCEPTS_FILE}"
    --loo_method "${LOO_METHOD}"
    --base_tau "${BASE_TAU}"
    --harm_gamma_start "${HARM_GAMMA_START}"
    --harm_gamma_end "${HARM_GAMMA_END}"
    --topk_min "${TOPK_MIN}"
    --redistribute_mode "${REDISTRIBUTE_MODE}"
  )
fi

if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
  ARGS+=( --adaptive_threshold --central_percentile "${CENTRAL_PERCENTILE}" --tau_factor "${TAU_FACTOR}" )
fi

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

[ "${DEBUG}" = true ] && ARGS+=(--debug)
[ "${DEBUG_PROMPTS}" = true ] && ARGS+=(--debug_prompts)
[ "${DEBUG_STEPS}" = true ] && ARGS+=(--debug_steps)

echo "[INFO] Starting generation with Prob-Redistribution..."
nohup python generate_adaptive_loo.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
