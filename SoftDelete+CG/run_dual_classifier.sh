#!/bin/bash
# Dual Classifier Generation: 3-class monitoring + 4-class guidance
#
# 3-class: monitors P(harm) using softmax probability
# 4-class: computes spatial guidance gradient
#
# This avoids the "classifier fooling" problem where guidance gradient
# makes the same classifier think the latent is safe when it's not.

set -e

# Classifiers
CLASSIFIER_3CLASS="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLASSIFIER_4CLASS="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

# Stable Diffusion checkpoint
SD_CKPT="CompVis/stable-diffusion-v1-4"

# Prompt file
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity.csv"

# Parameters
MONITORING_THRESHOLD=0.5
GUIDANCE_SCALE=10.0
BASE_GUIDANCE_SCALE=2.0
SPATIAL_THRESHOLD_START=0.5
SPATIAL_THRESHOLD_END=0.1
NUM_STEPS=50
SEED=42

# Output
OUTPUT_DIR="./scg_outputs/dual_classifier_test"

echo "=============================================="
echo "DUAL CLASSIFIER GENERATION"
echo "=============================================="
echo "3-class (monitoring): ${CLASSIFIER_3CLASS}"
echo "4-class (guidance):   ${CLASSIFIER_4CLASS}"
echo "Monitoring threshold: ${MONITORING_THRESHOLD}"
echo "Guidance scale:       ${GUIDANCE_SCALE}"
echo "=============================================="

python generate_nudity_dual_classifier.py \
    --ckpt_path "${SD_CKPT}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_3class_ckpt "${CLASSIFIER_3CLASS}" \
    --classifier_4class_ckpt "${CLASSIFIER_4CLASS}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --monitoring_threshold ${MONITORING_THRESHOLD} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --num_inference_steps ${NUM_STEPS} \
    --seed ${SEED} \
    --nsamples 1 \
    --debug

echo "Done! Output saved to ${OUTPUT_DIR}"
