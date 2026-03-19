#!/bin/bash
# ============================================================================
# GradCAM Visualization Script
# ============================================================================

export CUDA_VISIBLE_DEVICES=0
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# === Option 1: Single image ===
# python visualize_gradcam_heatmap.py \
#     --image_path /path/to/image.jpg \
#     --classifier_ckpt "${CLASSIFIER_CKPT}" \
#     --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
#     --pretrained_model "${SD_MODEL}" \
#     --target_class 2 \
#     --timestep 500 \
#     --output_dir ./gradcam_vis

# === Option 2: Directory of images ===
# Nudity images
python visualize_gradcam_heatmap.py \
    --image_dir /mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k \
    --num_images 10 \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --pretrained_model "${SD_MODEL}" \
    --target_class 2 \
    --timestep 500 \
    --output_dir ./gradcam_vis/nudity_t500

# Also check at different timesteps
for T in 100 300 700 900; do
    python visualize_gradcam_heatmap.py \
        --image_dir /mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k \
        --num_images 5 \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --pretrained_model "${SD_MODEL}" \
        --target_class 2 \
        --timestep ${T} \
        --output_dir ./gradcam_vis/nudity_t${T}
done

echo ""
echo "=============================================="
echo "Visualization complete!"
echo "Check results in: ./gradcam_vis/"
echo "=============================================="
