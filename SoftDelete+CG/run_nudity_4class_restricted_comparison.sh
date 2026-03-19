#!/bin/bash
# ============================================================================
# Nudity 4-Class Restricted Gradient Comparison
#
# Runs all variants in the same folder for easy comparison:
#   1. baseline (no restricted) - 기존 always 버전
#   2. safe_harm_restricted - safe와 harm 간 restricted
#   3. harm_harm_restricted - harm_nude와 harm_color 간 restricted
#   4. 3way_restricted - safe, harm_nude, harm_color 모두 restricted
#   5. nude_only - harm_color 무시, harm_nude만 guidance
#   6. 3class_spatial - safe+harm_nude+harm_color 모두 spatial (no restricted)
#   7. prob_threshold - prob > 0.2인 harm class만 guidance (skip 가능)
#
# All outputs go to: scg_outputs/nudity_4class_restricted_comparison/
# ============================================================================

export CUDA_VISIBLE_DEVICES=6

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ============================================================================
# Fixed parameters (best config from noskip)
# ============================================================================
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
PROMPT_FILE="./prompts/sexual_50.txt"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

BASE_OUTPUT_DIR="./scg_outputs/nudity_4class_restricted_comparison"

NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234
GUIDANCE_SCALE=10.0
SPATIAL_THRESHOLD_START=0.5
SPATIAL_THRESHOLD_END=0.1
HARMFUL_SCALE=1.0
BASE_GUIDANCE_SCALE=2.0
THRESHOLD_STRATEGY="cosine_anneal"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

mkdir -p "${BASE_OUTPUT_DIR}"

echo "=============================================="
echo "NUDITY 4-CLASS RESTRICTED GRADIENT COMPARISON"
echo "=============================================="
echo "Output: ${BASE_OUTPUT_DIR}"
echo "Config: gs${GUIDANCE_SCALE}_thr${SPATIAL_THRESHOLD_START}-${SPATIAL_THRESHOLD_END}_hs${HARMFUL_SCALE}_bgs${BASE_GUIDANCE_SCALE}"
echo "=============================================="

# ============================================================================
# 1. BASELINE (no restricted gradient)
# ============================================================================
echo ""
echo "[1/7] Running BASELINE (no restricted)..."
python generate_nudity_4class_spatial_cg_always.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/baseline" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: baseline"

# ============================================================================
# 2. SAFE-HARM RESTRICTED
# ============================================================================
echo ""
echo "[2/7] Running SAFE-HARM RESTRICTED..."
python generate_nudity_4class_spatial_cg_always_safe_harm_restricted.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/safe_harm_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: safe_harm_restricted"

# ============================================================================
# 3. HARM-HARM RESTRICTED
# ============================================================================
echo ""
echo "[3/7] Running HARM-HARM RESTRICTED..."
python generate_nudity_4class_spatial_cg_always_restricted.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/harm_harm_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: harm_harm_restricted"

# ============================================================================
# 4. 3-WAY RESTRICTED (safe + harm_nude + harm_color all orthogonalized)
# ============================================================================
echo ""
echo "[4/7] Running 3-WAY RESTRICTED..."
python generate_nudity_4class_spatial_cg_always_3way_restricted.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/3way_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: 3way_restricted"

# ============================================================================
# 5. NUDE ONLY (ignore harm_color, only guide harm_nude)
# ============================================================================
echo ""
echo "[5/7] Running NUDE ONLY..."
python generate_nudity_4class_spatial_cg_always_nude_only.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/nude_only" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: nude_only"

# ============================================================================
# 6. 3-CLASS SPATIAL (safe + harm_nude + harm_color, NO restricted)
# ============================================================================
echo ""
echo "[6/7] Running 3-CLASS SPATIAL (no restricted)..."
python generate_nudity_4class_spatial_cg_always_3class_spatial.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/3class_spatial" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP}

echo "Completed: 3class_spatial"

# ============================================================================
# 7. PROB THRESHOLD (prob > 0.2 -> guide, else skip)
# ============================================================================
echo ""
echo "[7/7] Running PROB THRESHOLD (prob > 0.2)..."
python generate_nudity_4class_spatial_cg_prob_threshold.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/prob_threshold_0.2" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP} \
    --harm_prob_threshold 0.2

echo "Completed: prob_threshold_0.2"

echo ""
echo "=============================================="
echo "ALL COMPLETE!"
echo "=============================================="
echo "Results: ${BASE_OUTPUT_DIR}/"
echo "  - baseline/"
echo "  - safe_harm_restricted/"
echo "  - harm_harm_restricted/"
echo "  - 3way_restricted/"
echo "  - nude_only/"
echo "  - 3class_spatial/"
echo "  - prob_threshold_0.2/"
echo "=============================================="
