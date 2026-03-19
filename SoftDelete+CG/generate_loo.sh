#!/usr/bin/env bash
set -euo pipefail

# =========================
# Machine Unlearning with LEAVE-ONE-OUT (LOO) Token Importance
# Revolutionary approach: Measure token criticality via embedding impact
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
# PROMPT_FILE="./prompts/sexual_50.txt"
PROMPT_FILE="./prompts/country_nude_body.txt"

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./SDCGlooMM/CNB"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Harmful Concept Suppression with LOO
# =========================
HARM_SUPPRESS=true
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"

# =========================
# LOO MODE SELECTION ⭐
# =========================
# Options:
#   - "cosine": Original method (cosine similarity only)
#   - "criticality": LOO criticality only (ignores cosine similarity)
#   - "combined": Multiply cosine × criticality^weight (RECOMMENDED!)
LOO_MODE="combined"

# Criticality weight (only used in "combined" mode)
# Score = cosine_similarity × (criticality^weight)
# Higher weight = more emphasis on criticality
# Recommended: 1.0 (balanced), 0.5 (less emphasis), 2.0 (more emphasis)
CRITICALITY_WEIGHT=1.0

# Normalize criticality scores to [0, 1] range
# Recommended: true (makes scores comparable across prompts)
CRITICALITY_NORMALIZE=true

# =========================
# ADAPTIVE THRESHOLD
# =========================
ADAPTIVE_THRESHOLD=true          # Enable adaptive threshold
BASE_TAU=0.15                    # Base threshold (fallback)
CENTRAL_PERCENTILE=0.80          # Use central 80% for mean
TAU_FACTOR=1.05                  # Multiplicative factor

# Gamma schedule (suppression strength over time)
HARM_GAMMA_START=40.0
HARM_GAMMA_END=0.5

# =========================
# Classifier Guidance
# =========================
CLASSIFIER_GUIDANCE=true
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
GUIDANCE_SCALE=7.5
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = clothed people

# =========================
# DEBUG Options
# =========================
DEBUG=true              # Enable general debug
DEBUG_PROMPTS=true      # Show LOO criticality analysis per token
DEBUG_STEPS=true        # Show per-step suppression stats

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_loo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║           MACHINE UNLEARNING - LEAVE-ONE-OUT (LOO) TOKEN IMPORTANCE            ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""
echo "┌─ LOO Algorithm ───────────────────────────────────────────────────────────────┐"
echo "│  Innovation: Measure token CRITICALITY via Leave-One-Out embedding impact"
echo "│  "
echo "│  For each token i in prompt:"
echo "│    1. Compute full embedding E_full"
echo "│    2. Compute embedding without token i: E_-i"
echo "│    3. Criticality_i = ||E_full - E_-i||  (L2 distance)"
echo "│  "
echo "│  → Critical tokens (e.g., 'nude') have HIGH criticality"
echo "│  → Non-critical tokens (e.g., 'a', 'the') have LOW criticality"
echo "│  "
echo "│  Suppression score = cosine_similarity × criticality^weight"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ LOO Configuration ───────────────────────────────────────────────────────────┐"
echo "│  Enabled: ${HARM_SUPPRESS}"
echo "│  Concepts file: ${HARM_CONCEPTS_FILE}"
echo "│  "
echo "│  LOO Mode: ${LOO_MODE}"
if [ "${LOO_MODE}" = "combined" ]; then
    echo "│    - Score = cosine × (criticality^${CRITICALITY_WEIGHT})"
    echo "│    - Criticality normalization: ${CRITICALITY_NORMALIZE}"
elif [ "${LOO_MODE}" = "cosine" ]; then
    echo "│    - Original method (cosine similarity only)"
elif [ "${LOO_MODE}" = "criticality" ]; then
    echo "│    - Pure LOO (criticality only, no cosine)"
fi
echo "│  "
if [ "${ADAPTIVE_THRESHOLD}" = true ]; then
    echo "│  Threshold Mode: ADAPTIVE ⭐"
    echo "│    - Base τ (fallback): ${BASE_TAU}"
    echo "│    - Central percentile: ${CENTRAL_PERCENTILE}"
    echo "│    - Multiplicative factor: ×${TAU_FACTOR}"
    echo "│    - Formula: τ_adaptive = central_mean × ${TAU_FACTOR}"
else
    echo "│  Threshold Mode: FIXED"
    echo "│    - Fixed τ: ${BASE_TAU}"
fi
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
echo "│  Per-token LOO analysis: ${DEBUG_PROMPTS}"
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

# Add LOO harmful concept suppression arguments
if [ "${HARM_SUPPRESS}" = true ]; then
    ARGS+=(
        --harm_suppress
        --harm_concepts_file "${HARM_CONCEPTS_FILE}"
        --base_tau "${BASE_TAU}"
        --harm_gamma_start "${HARM_GAMMA_START}"
        --harm_gamma_end "${HARM_GAMMA_END}"
        --loo_mode "${LOO_MODE}"
        --criticality_weight "${CRITICALITY_WEIGHT}"
    )

    if [ "${CRITICALITY_NORMALIZE}" = true ]; then
        ARGS+=(--criticality_normalize)
    fi
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

echo "[INFO] Starting LOO generation..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_loo.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "║"
echo "║  LOO Innovation Summary:"
echo "║    - Measures token criticality via embedding impact"
echo "║    - Smart suppression: High criticality + High similarity → Suppress"
echo "║    - Ignores non-critical tokens even if they match harmful concepts"
echo "║    - Adaptive threshold per prompt for robustness"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
