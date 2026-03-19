#!/bin/bash
# ============================================================================
# Generate images using Adaptive Spatial CG with SKIP MODE
# (Best Config from GPT-4o eval)
#
# Best Config: gs15.0_hs1.5_st0.1-0.4_cosine_anneal
# - guidance_scale: 15.0
# - harmful_scale: 1.5
# - spatial_threshold: 0.1 → 0.4
# - strategy: cosine_anneal
# - GPT-4o Results: Safe=39, Partial=10, Full=0 (98% Success)
#
# Uses: generate_adaptive_spatial_cg.py (3class classifier, SKIP mode)
# - Skip mode: only apply guidance when harmful content detected
#
# Usage: bash generate_nudity_adaptive_spatial_cg.sh <GPU> <PROMPT_FILE> <OUTPUT_DIR>
# Example: bash generate_nudity_adaptive_spatial_cg.sh 0 prompts/nudity.txt outputs/nudity_adaptive
# ============================================================================

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <GPU> <PROMPT_FILE> <OUTPUT_DIR>"
    echo ""
    echo "Example:"
    echo "  $0 0 /path/to/prompts.txt /path/to/output"
    exit 1
fi

GPU=$1
PROMPT_FILE=$2
OUTPUT_DIR=$3

export CUDA_VISIBLE_DEVICES=${GPU}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# ============================================================================
# Best Config Parameters (from GPT-4o evaluation)
# gs15.0_hs1.5_st0.1-0.4_cosine_anneal
# ============================================================================
SEED=42
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# Adaptive Spatial CG params
GUIDANCE_SCALE=15.0
HARMFUL_SCALE=1.5
SPATIAL_THR_START=0.1
SPATIAL_THR_END=0.4
THRESHOLD_STRATEGY="cosine_anneal"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Skip mode params
HARMFUL_THRESHOLD=0.5

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "Adaptive Spatial CG Generation (3class, SKIP)"
echo "=============================================="
echo "GPU: ${GPU}"
echo "Prompt: ${PROMPT_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters (Best Config from GPT-4o):"
echo "  guidance_scale: ${GUIDANCE_SCALE}"
echo "  harmful_scale: ${HARMFUL_SCALE}"
echo "  spatial_threshold: ${SPATIAL_THR_START} → ${SPATIAL_THR_END}"
echo "  strategy: ${THRESHOLD_STRATEGY}"
echo "  classifier: nudity_three_class (step_11800)"
echo "  mode: SKIP (only guide when harmful detected)"
echo "  harmful_threshold: ${HARMFUL_THRESHOLD}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"
cd "${BASE_DIR}/SoftDelete+CG"

python generate_adaptive_spatial_cg.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_classes 3 \
    --harmful_class 2 \
    --safe_class 1 \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THR_START} \
    --spatial_threshold_end ${SPATIAL_THR_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP} \
    --harmful_scale ${HARMFUL_SCALE} \
    --harmful_threshold ${HARMFUL_THRESHOLD} \
    --use_bidirectional \
    --skip_safe

echo ""
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="
