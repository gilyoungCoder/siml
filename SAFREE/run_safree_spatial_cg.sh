#!/bin/bash
# SAFREE++ Spatial Classifier Guidance
# Combines SAFREE text projection + Spatial CG image guidance

export CUDA_VISIBLE_DEVICES=7

cd /mnt/home/yhgil99/unlearning/SAFREE

# ========================================
# Configuration
# ========================================

# Model
MODEL_PATH="CompVis/stable-diffusion-v1-4"

# Prompts
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/sexual_top50.txt"

# Classifier
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"

# GradCAM statistics
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"

# Output
OUTPUT_DIR="./results/safree_spatial_cg_nudity"

# ========================================
# SAFREE Parameters
# ========================================
SAFREE_ALPHA=0.01
SVF_UP_T=10
CATEGORY="nudity"

# ========================================
# Spatial CG Parameters
# ========================================
CG_GUIDANCE_SCALE=5.0
SPATIAL_THRESHOLD_START=0.7
SPATIAL_THRESHOLD_END=0.3
THRESHOLD_STRATEGY="linear_decrease"
HARMFUL_SCALE=1.0
BASE_GUIDANCE_SCALE=0.0

# ========================================
# Generation Parameters
# ========================================
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
SEED=42
NSAMPLES=1

# ========================================
# Run SAFREE++ Spatial CG
# ========================================
GRADCAM_OPT=""
if [ -n "${GRADCAM_STATS_DIR}" ]; then
    GRADCAM_OPT="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
fi

python generate_safree_spatial_cg.py \
    --ckpt_path "${MODEL_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    ${GRADCAM_OPT} \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --safree \
    --safree_alpha ${SAFREE_ALPHA} \
    --svf \
    --svf_up_t ${SVF_UP_T} \
    --category "${CATEGORY}" \
    --spatial_cg \
    --cg_guidance_scale ${CG_GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step 0 \
    --guidance_end_step ${NUM_INFERENCE_STEPS} \
    --debug
