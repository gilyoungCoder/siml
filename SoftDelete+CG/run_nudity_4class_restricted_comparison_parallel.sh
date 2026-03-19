#!/bin/bash
# ============================================================================
# Nudity 4-Class Restricted Gradient Comparison - PARALLEL VERSION
#
# Runs all 7 variants in PARALLEL on different GPUs
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ============================================================================
# Fixed parameters
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
echo "PARALLEL EXECUTION - 7 experiments on 7 GPUs"
echo "=============================================="

# 1. BASELINE (GPU 0)
CUDA_VISIBLE_DEVICES=0 python generate_nudity_4class_spatial_cg_always.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/baseline" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 0] baseline started"

# 2. SAFE-HARM RESTRICTED (GPU 1)
CUDA_VISIBLE_DEVICES=1 python generate_nudity_4class_spatial_cg_always_safe_harm_restricted.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/safe_harm_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 1] safe_harm_restricted started"

# 3. HARM-HARM RESTRICTED (GPU 2)
CUDA_VISIBLE_DEVICES=2 python generate_nudity_4class_spatial_cg_always_restricted.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/harm_harm_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 2] harm_harm_restricted started"

# 4. 3-WAY RESTRICTED (GPU 3)
CUDA_VISIBLE_DEVICES=3 python generate_nudity_4class_spatial_cg_always_3way_restricted.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/3way_restricted" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 3] 3way_restricted started"

# 5. NUDE ONLY (GPU 4)
CUDA_VISIBLE_DEVICES=4 python generate_nudity_4class_spatial_cg_always_nude_only.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/nude_only" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --use_bidirectional --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 4] nude_only started"

# 6. 3-CLASS SPATIAL (GPU 5)
CUDA_VISIBLE_DEVICES=5 python generate_nudity_4class_spatial_cg_always_3class_spatial.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/3class_spatial" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} &
echo "[GPU 5] 3class_spatial started"

# 7. PROB THRESHOLD (GPU 6)
CUDA_VISIBLE_DEVICES=6 python generate_nudity_4class_spatial_cg_prob_threshold.py \
    "${SD_MODEL}" --prompt_file "${PROMPT_FILE}" --output_dir "${BASE_OUTPUT_DIR}/prob_threshold_0.2" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} --cfg_scale ${CFG_SCALE} --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} --guidance_scale ${GUIDANCE_SCALE} --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} --threshold_strategy "${THRESHOLD_STRATEGY}" \
    --harmful_scale ${HARMFUL_SCALE} --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step ${GUIDANCE_START_STEP} --guidance_end_step ${GUIDANCE_END_STEP} \
    --harm_prob_threshold 0.2 &
echo "[GPU 6] prob_threshold_0.2 started"

echo ""
echo "=============================================="
echo "All 7 experiments launched in parallel!"
echo "Waiting for completion..."
echo "=============================================="

wait

echo ""
echo "=============================================="
echo "ALL COMPLETE!"
echo "=============================================="
echo "Results: ${BASE_OUTPUT_DIR}/"
ls -la "${BASE_OUTPUT_DIR}/"
echo "=============================================="
