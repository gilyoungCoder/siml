#!/bin/bash
# ============================================================================
# Train z0-space 4-class ResNet18 classifier
#
# Classes:
#   0: benign (no people) — imagenet5k (4805 images)
#   1: safe_clothed (person with clothes) — safe + safe_failure (2400 images)
#   2: harm_nude (nudity) — Wnudity5k (4900 images)
#   3: harm_color (color artifacts) — color_artifacts_strong (1200 images)
#
# With balance_classes, all classes undersampled to 1200.
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

LOG_FILE="./work_dirs/z0_resnet18_4class/train.log"
mkdir -p ./work_dirs/z0_resnet18_4class

nohup python train_4class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_data_path "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --person_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe" \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe_failure" \
    --nudity_data_path "/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k" \
    --harm_color_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/color_artifacts_strong" \
    --output_dir "./work_dirs/z0_resnet18_4class" \
    --num_classes 4 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_train_steps 20000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --use_wandb \
    --wandb_project "z0_classifier" \
    --wandb_run_name "z0_resnet18_4class" \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
