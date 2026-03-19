#!/bin/bash
# [Z0] GPU 1 — violence 9-class
export CUDA_VISIBLE_DEVICES=1

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

nohup python train_i2p_9class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_dir "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --data_dir "/mnt/home/yhgil99/dataset/threeclassImg/i2p/violence_8class" \
    --concept_name "violence" \
    --output_dir "./work_dirs/z0_violence_9class" \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 25000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --mixed_precision no \
    --use_wandb \
    --wandb_project "z0_i2p_9class_classifier" \
    --wandb_run_name "z0_violence_9class" \
    > logs/train_violence.log 2>&1 &

echo "[GPU 1] violence 9-class started. PID=$!"
