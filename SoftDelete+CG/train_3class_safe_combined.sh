#!/bin/bash
# ============================================================================
# Train 3-class classifier with combined safe directories
# Person class: safe + safe_failure (half from each)
# Grayscale: OFF (color images)
# ============================================================================

export CUDA_VISIBLE_DEVICES=7

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

python train_3class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_data_path "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --person_data_path "/mnt/home/yhgil99/dataset/threeclassImg/People5k" \
    --nudity_data_path "/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k" \
    --output_dir "./work_dirs/nudity_three_class_safe_combined_legacy" \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 20000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --use_wandb \
    --report_to wandb \
    --wandb_project "three_class_classifier" \
    --wandb_run_name "nudity_safe_combined_legacy"
