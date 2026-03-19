#!/bin/bash
# ============================================================================
# Nudity Classifier Training with Color Augmentation
# Purpose: Reduce color bias by adding ColorJitter and RandomGrayscale
# ============================================================================

export CUDA_VISIBLE_DEVICES=4
export TORCH_XLA_DISABLE_FLASH_ATTENTION=1

# ============================================================================
# Configuration
# ============================================================================

output_dir=nudity_three_class_color_aug
benign_data_path=/mnt/home/yhgil99/dataset/benign_data
nudity_data_path=/mnt/home/yhgil99/dataset/nudity  # Update this path to your nudity dataset
image_resolution=512
pretrained_model_mode=stablediffusion
seen_dataset_ratio=1.0  # Use 100% of data
learning_rate=1.0e-4
noise_std=1.0

# Training parameters
train_batch_size=16
save_ckpt_freq=200  # Save checkpoint every 200 steps
num_train_epochs=20  # Adjust as needed

# Pretrained model
if [ $pretrained_model_mode == "stablediffusion" ]; then
    pretrained_model=CompVis/stable-diffusion-v1-4
fi

# ============================================================================
# Build command
# ============================================================================

command_line="python ../three_classificaiton/train_nudity_classifier.py "
command_line+="--benign_data_path $benign_data_path "
command_line+="--nudity_data_path $nudity_data_path "
command_line+="--output_dir work_dirs/$output_dir "
command_line+="--report_to wandb "
command_line+="--use_wandb "
command_line+="--train_batch_size $train_batch_size "
command_line+="--pretrained_model_name_or_path $pretrained_model "
command_line+="--save_ckpt_freq $save_ckpt_freq "
command_line+="--learning_rate $learning_rate "
command_line+="--noise_std $noise_std "
command_line+="--num_train_epochs $num_train_epochs "

echo "============================================================================"
echo "Training Nudity Classifier with Color Augmentation"
echo "============================================================================"
echo "Output directory: work_dirs/$output_dir"
echo "Benign data: $benign_data_path"
echo "Nudity data: $nudity_data_path"
echo "Batch size: $train_batch_size"
echo "Epochs: $num_train_epochs"
echo "Learning rate: $learning_rate"
echo ""
echo "Color Augmentation:"
echo "  - ColorJitter: brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1"
echo "  - RandomGrayscale: p=0.1"
echo "============================================================================"
echo ""
echo "Executing: $command_line"
echo ""

# Run in background and save logs
nohup $command_line > train_nudity_color_aug.log 2>&1 &

echo "Training started in background. Check logs at: train_nudity_color_aug.log"
echo "PID: $!"

exit 0
