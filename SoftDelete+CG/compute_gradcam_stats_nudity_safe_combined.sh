#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Nudity Safe Combined (step_18900)
# ============================================================================

export CUDA_VISIBLE_DEVICES=7

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --classifier_ckpt ./work_dirs/nudity_three_class_safe_combined/checkpoint/step_18900/classifier.pth \
    --output_file ./gradcam_nudity_stats_safe_combined.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_nudity_stats_safe_combined.json"
