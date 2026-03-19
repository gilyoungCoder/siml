#!/bin/bash
# ============================================================================
# Train 9-class classifiers for all I2P concepts (except sexual)
#
# 9-class structure:
#   0: benign (imagenet)
#   1: harm0, 2: safe0
#   3: harm1, 4: safe1
#   5: harm2, 6: safe2
#   7: harm3, 8: safe3
#
# Concepts: harassment, hate, illegal, selfharm, shocking, violence
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Common settings
SD_MODEL="CompVis/stable-diffusion-v1-4"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
I2P_BASE="/mnt/home/yhgil99/dataset/threeclassImg/i2p"

TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=25000
SAVE_CKPT_FREQ=100
SEED=42

# List of concepts (excluding sexual)
CONCEPTS=("harassment" "hate" "illegal" "selfharm" "shocking" "violence")

for concept in "${CONCEPTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training 9-class classifier for: ${concept}"
    echo "=============================================="
    echo ""

    DATA_DIR="${I2P_BASE}/${concept}_8class"

    python train_i2p_9class.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --benign_dir "${BENIGN_DIR}" \
        --data_dir "${DATA_DIR}" \
        --concept_name "${concept}" \
        --output_dir "./work_dirs/${concept}_9class" \
        --train_batch_size ${TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --max_train_steps ${MAX_TRAIN_STEPS} \
        --save_ckpt_freq ${SAVE_CKPT_FREQ} \
        --seed ${SEED} \
        --mixed_precision no \
        --use_wandb \
        --report_to wandb \
        --wandb_project "i2p_9class_classifier" \
        --wandb_run_name "${concept}_9class"

    echo ""
    echo "Completed: ${concept}"
    echo ""
done

echo ""
echo "=============================================="
echo "ALL CONCEPTS TRAINING COMPLETE!"
echo "=============================================="
