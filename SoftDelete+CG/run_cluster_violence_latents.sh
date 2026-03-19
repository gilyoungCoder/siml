#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# Violence Latent Clustering Script
# Mode-Aware Guidance를 위한 cluster centroids 생성
#───────────────────────────────────────────────────────────────────────────────#

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/mnt/home/yhgil99/unlearning/SoftDelete+CG"

# Mode 선택: "generate" 또는 "fit"
MODE="generate"

# Model paths
CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"

# Input options
PROMPTS_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/violence/harm.txt"  # violence prompt 목록 파일
# LATENTS_PATH="collected_violence_latents.pt"  # fit 모드 시 사용

# Generation options
NUM_SAMPLES_PER_PROMPT=10
CAPTURE_TIMESTEPS="900 800 700 600 500 400 300 200 100"  # 9 timesteps
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5
HARMFUL_CLASS=2
CONFIDENCE_THRESHOLD=0.3

# Clustering options
N_CLUSTERS=10
POOLING="mean"

# Output
OUTPUT_PATH="cluster_centroids/violence_clusters.pt"

# 실행
nohup python mode_aware_guidance/cluster_harmful_latents.py \
    --mode $MODE \
    --ckpt_path "$CKPT_PATH" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --prompts_file "$PROMPTS_FILE" \
    --num_samples_per_prompt $NUM_SAMPLES_PER_PROMPT \
    --capture_timesteps $CAPTURE_TIMESTEPS \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --harmful_class $HARMFUL_CLASS \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --n_clusters $N_CLUSTERS \
    --pooling $POOLING \
    --output_path "$OUTPUT_PATH" \
    --save_latents \
    > cluster_violence.log 2>&1 &

echo "========================================"
echo "Clustering launched in background!"
echo "Output: $OUTPUT_PATH"
echo "Log: cluster_violence.log"
echo ""
echo "Monitor with: tail -f cluster_violence.log"
echo "========================================"
