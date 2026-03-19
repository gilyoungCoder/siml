#!/bin/bash
# ============================================================================
# Nudity 6-Class v2 Spatial Classifier Guidance Generation
#
# v2 Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes) - common safe target
#   2: harm_nude (completely naked + topless merged)
#   3: harm_lingerie
#   4: harm_swimwear
#   5: harm_color (color artifacts) - NEW
#
# Guidance: all harm (2-5) -> safe_clothed (1)
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Paths
MODEL_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/nudity_6class_v2/checkpoint/step_22700/classifier.pth"
PROMPT_FILE="./prompts/nudity_eval.txt"
OUTPUT_DIR="./scg_outputs/nudity_6class_v2_step22700"

# GradCAM stats directory (for absolute normalization)
GRADCAM_STATS_DIR="./gradcam_stats/nudity_6class_v2_step22700"

# Guidance parameters
GUIDANCE_SCALE=10.0
SPATIAL_THR_START=0.5
SPATIAL_THR_END=0.3
THR_STRATEGY="cosine_anneal"
HARMFUL_SCALE=1.0
BASE_GUIDANCE_SCALE=1.0

python generate_nudity_6class_spatial_cg.py \
    "${MODEL_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_PATH}" \
    --num_classes 6 \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples 1 \
    --cfg_scale 7.5 \
    --num_inference_steps 50 \
    --seed 1234 \
    --guidance_scale ${GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THR_START} \
    --spatial_threshold_end ${SPATIAL_THR_END} \
    --threshold_strategy ${THR_STRATEGY} \
    --use_bidirectional \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --guidance_start_step 0 \
    --guidance_end_step 50
