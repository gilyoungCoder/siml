#!/bin/bash
# Train 9-class classifier for violence concept

export CUDA_VISIBLE_DEVICES=4

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

python train_i2p_9class.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --benign_dir "/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k" \
    --data_dir "/mnt/home/yhgil99/dataset/threeclassImg/i2p/violence_8class" \
    --concept_name "violence" \
    --output_dir "./work_dirs/violence_9class" \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_train_steps 25000 \
    --save_ckpt_freq 100 \
    --seed 42 \
    --mixed_precision no \
    --use_wandb \
    --report_to wandb \
    --wandb_project "i2p_9class_classifier" \
    --wandb_run_name "violence_9class"
