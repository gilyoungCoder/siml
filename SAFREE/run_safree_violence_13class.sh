#!/bin/bash
# SAFREE++ Violence 13-Class Spatial CG

export CUDA_VISIBLE_DEVICES=6

cd /mnt/home/yhgil99/unlearning/SAFREE

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/violence_13class/checkpoint/step_28400/classifier.pth"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/violence_top50.txt"
OUTPUT_DIR="./results/safree_violence_13class"
# GradCAM stats (computed for step_28400, may not match current checkpoint)
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/violence_13class_step28400"

GRADCAM_OPT=""
if [ -d "${GRADCAM_STATS_DIR}" ]; then
    GRADCAM_OPT="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
fi

python generate_safree_violence_13class_spatial_cg.py \
    --ckpt_path "CompVis/stable-diffusion-v1-4" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    ${GRADCAM_OPT} \
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
