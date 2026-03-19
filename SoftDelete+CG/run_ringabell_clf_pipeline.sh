#!/bin/bash
# =============================================================================
# RingaBell Classifier Pipeline
# Step 1: Generate 1500 training images from RingaBell Full prompts (vanilla SD 1.4)
# Step 2: Retrain 4-class classifier with RingaBell nudity data
# =============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

GPU=${1:-0}
echo "Using GPU: $GPU"

# === Paths ===
TRAIN_CSV="ringabell_split/ringabell_train_full.csv"
NUDITY_DATA="/mnt/home/yhgil99/dataset/ringabell_nude_train"
WORK_DIR="./work_dirs/nudity_4class_ringabell"

# ============================================================
# STEP 1: Generate training images (vanilla SD 1.4)
# ============================================================
echo ""
echo "============================================"
echo "STEP 1: Generating 1500 training images"
echo "============================================"

if [ "$(ls -1 ${NUDITY_DATA}/*.png 2>/dev/null | wc -l)" -ge 1500 ]; then
    echo "SKIP: ${NUDITY_DATA} already has >= 1500 images"
else
    CUDA_VISIBLE_DEVICES=$GPU python generate_ringabell_train_images.py \
        --prompt_csv "$TRAIN_CSV" \
        --output_dir "$NUDITY_DATA" \
        --num_images_per_prompt 52 \
        --batch_size 4 \
        --cfg_scale 7.5 \
        --num_inference_steps 50 \
        --start_seed 0
fi

echo "Training images: $(ls -1 ${NUDITY_DATA}/*.png 2>/dev/null | wc -l)"

# ============================================================
# STEP 2: Train 4-class classifier
# ============================================================
echo ""
echo "============================================"
echo "STEP 2: Training 4-class classifier"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU python train_4class_safe_combined.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_data_path "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --person_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe" \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/safe_failure" \
    --nudity_data_path "$NUDITY_DATA" \
    --harm_color_data_path \
        "/mnt/home/yhgil99/dataset/threeclassImg/nudity/color_artifacts_strong" \
    --output_dir "$WORK_DIR" \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 20000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --use_wandb \
    --report_to wandb \
    --wandb_project "four_class_classifier" \
    --wandb_run_name "nudity_4class_ringabell"

echo ""
echo "============================================"
echo "DONE"
echo "============================================"
echo "Training data: $NUDITY_DATA"
echo "Classifier: $WORK_DIR"
echo "Test prompts: ringabell_split/ringabell_test.csv"
