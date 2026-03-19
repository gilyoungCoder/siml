#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for VanGogh
# ============================================================================

export CUDA_VISIBLE_DEVICES=5
# --classifier_ckpt ./work_dirs/nudity_three_class_color_aug/checkpoint/step_16000/classifier.pth \

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/nudity/harm \
    --classifier_ckpt ./work_dirs/nudity_three_class_colored/checkpoint/step_11600/classifier.pth \
    --output_file ./gradcam_nudity_stats_colored.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_nudity_stats_colored.json"
