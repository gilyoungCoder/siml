#!/bin/bash
# ============================================================================
# Evaluate JTT Stage2 Classifier Accuracy
# step_18400 checkpoint accuracy 확인
# ============================================================================

export CUDA_VISIBLE_DEVICES=2

# Classifier checkpoint
CLASSIFIER_CKPT="./work_dirs/nudity_three_class_grayscale/checkpoint/step_11200/classifier.pth"

# Test data directories
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDE_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"

# Output directory
OUTPUT_DIR="./eval_results/jtt_stage2_step18400"

python eval_classifier_accuracy.py \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --benign_dir "$BENIGN_DIR" \
    --person_dir "$PERSON_DIR" \
    --nude_dir "$NUDE_DIR" \
    --batch_size 32 \
    --max_samples_per_class 500 \
    --timesteps "0,100,250,500,750" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Results saved to: $OUTPUT_DIR"
