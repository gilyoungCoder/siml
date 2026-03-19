#!/bin/bash
# Machine Unlearning: Remove Nudity from Sexual Prompts using FK Steering

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

# Configuration
PROMPT_FILE="./prompts/sexual_50.txt"
# PROMPT_FILE="./prompts/country_nude_body.txt"

OUTPUT_DIR="./unlearned_outputs/i2psexual_fk_steering"
# OUTPUT_DIR="./unlearned_outputs/CNB_fk_steering"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
MODEL_ID="CompVis/stable-diffusion-v1-4"

# FK Steering Parameters
NUM_PARTICLES=4
POTENTIAL_TYPE="max"
LAMBDA_SCALE=15.0        # Higher = stronger nudity removal
RESAMPLING_INTERVAL=10

# Generation Parameters
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5
NSAMPLES=1
SEED=42

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    MACHINE UNLEARNING - NUDITY REMOVAL                         ║"
echo "║                          with FK Steering                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Input:  Prompts with sexual/nude content"
echo "Output: Images with nudity REMOVED (steered to clothed people)"
echo ""
echo "┌─ Configuration ───────────────────────────────────────────────────────────────┐"
echo "│  Prompt file: ${PROMPT_FILE}"
echo "│  Output dir:  ${OUTPUT_DIR}"
echo "│  Classifier:  ${CLASSIFIER_CKPT}"
echo "│"
echo "│  FK Steering:"
echo "│    - Particles (k):     ${NUM_PARTICLES}"
echo "│    - Potential:         ${POTENTIAL_TYPE}"
echo "│    - Lambda:            ${LAMBDA_SCALE} (nudity removal strength)"
echo "│    - Resample interval: ${RESAMPLING_INTERVAL} steps"
echo "│"
echo "│  Generation:"
echo "│    - Steps:       ${NUM_INFERENCE_STEPS}"
echo "│    - CFG scale:   ${GUIDANCE_SCALE}"
echo "│    - Samples/prompt: ${NSAMPLES}"
echo "│    - Seed:        ${SEED}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/run.log"

# Run FK steering unlearning
echo "[INFO] Starting FK steering unlearning..."
echo "[INFO] Log: ${LOG}"
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --model_id "${MODEL_ID}" \
    --num_particles "${NUM_PARTICLES}" \
    --potential_type "${POTENTIAL_TYPE}" \
    --lambda_scale "${LAMBDA_SCALE}" \
    --resampling_interval "${RESAMPLING_INTERVAL}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}" \
    --generate_baseline \
    --verbose \
    2>&1 | tee "${LOG}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ Complete!"
echo "║"
echo "║  Results: ${OUTPUT_DIR}"
echo "║  Log:     ${LOG}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
