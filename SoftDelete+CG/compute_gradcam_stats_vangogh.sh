#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for VanGogh
# ============================================================================

export CUDA_VISIBLE_DEVICES=6

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/VanGogh/harm \
    --classifier_ckpt ./work_dirs/vangogh_three_class_diff/checkpoint/step_7200/classifier.pth \
    --output_file ./gradcam_vangogh_stats.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_vangogh_stats.json"
