#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Nudity (Grayscale Classifier)
# ============================================================================

export CUDA_VISIBLE_DEVICES=7

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/nudity/harm \
    --classifier_ckpt ./work_dirs/nudity_three_class_grayscale/checkpoint/step_11200/classifier.pth \
    --output_file ./gradcam_nudity_stats_grayscale.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_nudity_stats_grayscale.json"
