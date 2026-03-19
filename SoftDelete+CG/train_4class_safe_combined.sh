#!/bin/bash
# ============================================================================
# Train 4-class classifier with combined safe directories + harm_color
#
# Classes:
#   0: benign (no people)
#   1: safe_clothed (person with clothes - combined from safe + safe_failure)
#   2: harm_nude (nudity)
#   3: harm_color (normal images but with color artifacts/distortions)
#
# ============================================================================

export CUDA_VISIBLE_DEVICES=5

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

python train_4class_safe_combined.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_data_path "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --person_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe" \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe_failure" \
    --nudity_data_path "/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k" \
    --harm_color_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/color_artifacts_strong" \
    --output_dir "./work_dirs/nudity_4class_safe_combined" \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 20000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --use_wandb \
    --report_to wandb \
    --wandb_project "four_class_classifier" \
    --wandb_run_name "nudity_4class_safe_combined"
