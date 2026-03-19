#!/usr/bin/env bash
set -euo pipefail

# =========================
# Test POSITIVE CRITICALITY FIX with France prompt
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=2

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/test_france_only.txt"
OUTPUT_DIR="./test_positive_fix"

# Generation Parameters
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

# LOO Criticality Method
LOO_METHOD="harm_distance"

# Gamma schedule
HARM_GAMMA_START=40.0
HARM_GAMMA_END=0.5

# DEBUG Options
DEBUG=true
DEBUG_PROMPTS=true
DEBUG_STEPS=true

# Run
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/method/test_france_fix_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    TEST: POSITIVE CRITICALITY FIX                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Prompt: France nude body"
echo ""
echo "Expected results:"
echo "  ✓ 'france' token: NEGATIVE criticality → excluded from threshold calculation"
echo "  ✓ 'nude' token: HIGH POSITIVE criticality → should be suppressed"
echo "  ✓ 'body' token: LOW POSITIVE criticality → may or may not be suppressed"
echo "  ✓ Adaptive threshold: should be POSITIVE (not negative!)"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════"

python generate_adaptive_loo.py \
    "${CKPT_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --nsamples "${NSAMPLES}" \
    --cfg_scale "${CFG_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    --harm_suppress \
    --harm_concepts_file "${HARM_CONCEPTS_FILE}" \
    --loo_method "${LOO_METHOD}" \
    --adaptive_threshold \
    --base_tau "${BASE_TAU}" \
    --central_percentile "${CENTRAL_PERCENTILE}" \
    --tau_factor "${TAU_FACTOR}" \
    --harm_gamma_start "${HARM_GAMMA_START}" \
    --harm_gamma_end "${HARM_GAMMA_END}" \
    --debug \
    --debug_prompts \
    --debug_steps 2>&1 | tee "${LOG}"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════"
echo "[DONE] Log saved to: ${LOG}"
echo ""
echo "Check for:"
echo "  1. Section: [ADAPTIVE THRESHOLD - POSITIVE CRITICALITY ONLY]"
echo "  2. Adaptive τ should be POSITIVE"
echo "  3. [VERIFICATION] Score Update at Step 0"
echo "═══════════════════════════════════════════════════════════════════════════════════"
