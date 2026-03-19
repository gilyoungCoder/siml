#!/bin/bash
# ============================================================================
# Train 9-class violence classifier
# Classes:
#   0: benign (no people)
#   1: harm_fighting -> 2: safe_fighting
#   3: harm_weapon   -> 4: safe_weapon
#   5: harm_blood    -> 6: safe_blood
#   7: harm_war      -> 8: safe_war
# ============================================================================

export CUDA_VISIBLE_DEVICES=6

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Base directory for 9-class violence dataset
DATA_BASE="/mnt/home/yhgil99/dataset/threeclassImg/violence_9class"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"

python train_violence_9class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_dir "${BENIGN_DIR}" \
    --harm_fighting_dir "${DATA_BASE}/street_fight" \
    --safe_fighting_dir "${DATA_BASE}/friendly_handshake" \
    --harm_weapon_dir "${DATA_BASE}/attacking_with_knife" \
    --safe_weapon_dir "${DATA_BASE}/chopping_vegetables" \
    --harm_blood_dir "${DATA_BASE}/bloody_wounds" \
    --safe_blood_dir "${DATA_BASE}/bandaging_hospital" \
    --harm_war_dir "${DATA_BASE}/soldiers_battlefield" \
    --safe_war_dir "${DATA_BASE}/peace_conference" \
    --output_dir "./work_dirs/violence_9class" \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 30000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --mixed_precision no \
    --use_wandb \
    --report_to wandb \
    --wandb_project "violence_9class_classifier" \
    --wandb_run_name "violence_9class_v1"
