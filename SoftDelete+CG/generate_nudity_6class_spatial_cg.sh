#!/bin/bash
# ============================================================================
# Generate images with Nudity 6-Class Adaptive Spatial Classifier Guidance
#
# Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes) - target safe class
#   2: harm_full_nude (completely naked)
#   3: harm_topless (topless with jeans)
#   4: harm_lingerie (underwear/lingerie)
#   5: harm_swimwear (revealing bikini)
#
# Guidance: all harm (2-5) -> safe_clothed (1)
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Configuration
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/nudity_6class/checkpoint/step_28000/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_6class"
PROMPT_FILE="./prompts/nudity_prompts.txt"  # TODO: Set your prompt file
OUTPUT_DIR="./scg_outputs/nudity_6class_spatial_cg"

# Generation parameters
NUM_CLASSES=6
NSAMPLES=1
CFG_SCALE=7.5
NUM_STEPS=50
SEED=1234

# Guidance parameters
GUIDANCE_SCALE=5.0
HARMFUL_SCALE=1.0
BASE_GUIDANCE_SCALE=2.0
SPATIAL_THRESHOLD_START=0.7
SPATIAL_THRESHOLD_END=0.3
THRESHOLD_STRATEGY="linear_decrease"

# Active step range
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

echo "=============================================="
echo "Nudity 6-Class Spatial Classifier Guidance"
echo "=============================================="
echo "SD Model: ${SD_MODEL}"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "GradCAM Stats: ${GRADCAM_STATS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

python generate_nudity_6class_spatial_cg.py \
    "${SD_MODEL}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_classes ${NUM_CLASSES} \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_STEPS} \
    --seed ${SEED} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --harmful_scale ${HARMFUL_SCALE} \
    --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
    --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
    --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
    --threshold_strategy ${THRESHOLD_STRATEGY} \
    --guidance_start_step ${GUIDANCE_START_STEP} \
    --guidance_end_step ${GUIDANCE_END_STEP} \
    --use_bidirectional

echo ""
echo "=============================================="
echo "DONE! Images saved to: ${OUTPUT_DIR}"
echo "=============================================="
