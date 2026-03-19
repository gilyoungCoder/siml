#!/bin/bash
# ============================================================================
# Train 6-class nudity classifier
# Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes) - common safe target
#   2: harm_full_nude (completely naked)
#   3: harm_topless (topless with jeans)
#   4: harm_lingerie (underwear/lingerie)
#   5: harm_swimwear (revealing bikini)
#
# Guidance mapping: all harm (2-5) -> safe_clothed (1)
# ============================================================================

export CUDA_VISIBLE_DEVICES=5

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Data directories
DATA_BASE="/mnt/home/yhgil99/dataset/threeclassImg/nudity_6class"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"

python train_nudity_6class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_dir "${BENIGN_DIR}" \
    --clothed_dir "${DATA_BASE}/casual_clothes_aug" \
    --full_nude_dir "${DATA_BASE}/completely_naked" \
    --topless_dir "${DATA_BASE}/topless_with_jeans" \
    --lingerie_dir "${DATA_BASE}/lingerie" \
    --swimwear_dir "${DATA_BASE}/revealing_bikini" \
    --output_dir "./work_dirs/nudity_6class" \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 30000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --mixed_precision no \
    --use_wandb \
    --report_to wandb \
    --wandb_project "nudity_6class_classifier" \
    --wandb_run_name "nudity_6class_v1"
