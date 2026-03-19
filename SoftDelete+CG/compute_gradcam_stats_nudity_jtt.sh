#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Nudity using JTT Stage2 Classifier
# JTT stage2 step_18400 classifier를 사용한 GradCAM 통계 계산
# ============================================================================

export CUDA_VISIBLE_DEVICES=1

python compute_gradcam_statistics.py \
    --data_dir /mnt/home/yhgil99/dataset/threeclassImg/nudity/harm \
    --classifier_ckpt ./work_dirs/jtt_stage2_T5800/checkpoints/step_18400/classifier.pth \
    --output_file ./gradcam_nudity_stats_jtt_step18400.json \
    --num_samples 1000 \
    --harmful_class 2 \
    --gradcam_layer "encoder_model.middle_block.2"

echo "Done! Stats saved to gradcam_nudity_stats_jtt_step18400.json"
