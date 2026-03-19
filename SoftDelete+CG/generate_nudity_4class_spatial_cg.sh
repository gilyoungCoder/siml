#!/bin/bash
# ============================================================================
# Nudity 4-Class Selective Spatial Classifier Guidance
#
# Classes:
#   0: benign (no people)
#   1: safe_clothed (person with clothes)
#   2: harm_nude (nudity) - GUIDANCE APPLIED
#   3: harm_color (color artifacts) - GUIDANCE APPLIED
#
# Guidance Logic:
#   - If max logit is class 2 or 3 (harmful): apply spatial guidance
#   - If max logit is class 0 or 1 (safe): skip guidance
#   - Guidance direction: safe_grad - harm_grad (bidirectional)
#
# IMPORTANT: Run compute_gradcam_statistics_4class.sh FIRST to generate
#            GradCAM statistics before running this script!
#
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Classifier checkpoint (step 17100)
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"

# SD model path
SD_MODEL="CompVis/stable-diffusion-v1-4"

# Prompt file (adjust as needed)
PROMPT_FILE="./prompts/sexual_50.txt"

# Output directory
OUTPUT_DIR="./output_img/nudity_4class_selective_cg"

# GradCAM statistics directory (computed by compute_gradcam_statistics_4class.sh)
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

python generate_nudity_4class_spatial_cg.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples 1 \
    --cfg_scale 7.5 \
    --num_inference_steps 50 \
    --seed 1234 \
    --guidance_scale 5.0 \
    --spatial_threshold_start 0.7 \
    --spatial_threshold_end 0.3 \
    --threshold_strategy "linear_decrease" \
    --use_bidirectional \
    --harmful_scale 1.0 \
    --base_guidance_scale 0.0 \
    --guidance_start_step 0 \
    --guidance_end_step 50 \
    --debug \
    --save_visualizations
