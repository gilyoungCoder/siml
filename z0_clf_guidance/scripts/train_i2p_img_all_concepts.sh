#!/bin/bash
# ============================================================================
# [Z0-IMG] Launch ALL 8 image-space training jobs in parallel on GPU 0-7
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
mkdir -p logs

echo "Launching all z0 IMAGE-SPACE classifier training jobs..."
echo ""

# GPU 0: nudity 3-class
bash scripts/train_img.sh

# GPU 1-7: I2P concepts
CONCEPTS=("violence" "harassment" "hate" "illegal" "selfharm" "shocking" "sexual")
GPUS=(1 2 3 4 5 6 7)

for i in "${!CONCEPTS[@]}"; do
    concept="${CONCEPTS[$i]}"
    gpu="${GPUS[$i]}"

    CUDA_VISIBLE_DEVICES=${gpu} nohup python train_i2p_9class_img.py \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --benign_dir "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
        --data_dir "/mnt/home/yhgil99/dataset/threeclassImg/i2p/${concept}_8class" \
        --concept_name "${concept}" \
        --output_dir "./work_dirs/z0_img_${concept}_9class" \
        --train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --max_train_steps 25000 \
        --save_ckpt_freq 100 \
        --seed 42 \
        --mixed_precision no \
        --use_wandb \
        --wandb_project "z0_img_i2p_9class_classifier" \
        --wandb_run_name "z0_img_${concept}_9class" \
        > "logs/train_img_${concept}.log" 2>&1 &

    echo "[GPU ${gpu}] ${concept} 9-class (img) started. PID=$!"
done

echo ""
echo "=============================================="
echo "All 8 image-space jobs launched! Monitor with:"
echo "  tail -f logs/train_img_*.log"
echo "=============================================="
