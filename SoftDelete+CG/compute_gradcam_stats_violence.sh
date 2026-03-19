#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Violence
# ============================================================================

export CUDA_VISIBLE_DEVICES=5

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/violence_diffprompt/harm1k \
    --classifier_ckpt ./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth \
    --output_file ./gradcam_violence_stats.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_violence_stats.json"
