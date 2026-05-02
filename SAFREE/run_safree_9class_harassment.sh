#!/bin/bash
# SAFREE++ 9-Class Spatial CG for Harassment

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SAFREE

CONCEPT="harassment"
STEP="24300"
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/${CONCEPT}_9class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/${CONCEPT}_9class_step${STEP}"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/${CONCEPT}_top50.txt"
OUTPUT_DIR="./results/safree_9class_${CONCEPT}"

python generate_safree_9class_spatial_cg.py \
    --ckpt_path "CompVis/stable-diffusion-v1-4" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --category "${CONCEPT}" \
    --nsamples 1 \
    --cfg_scale 7.5 \
    --num_inference_steps 50 \
    --seed 42 \
    --safree \
    --safree_alpha 0.01 \
    --svf \
    --svf_up_t 10 \
    --spatial_cg \
    --cg_guidance_scale 5.0 \
    --spatial_threshold_start 0.7 \
    --spatial_threshold_end 0.3 \
    --threshold_strategy "linear_decrease" \
    --harmful_scale 1.0 \
    --base_guidance_scale 0.0 \
    --skip_if_safe \
    --debug
