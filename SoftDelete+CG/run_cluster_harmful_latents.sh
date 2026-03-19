#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# Harmful Latent Clustering Script
# Mode-Aware Guidance를 위한 cluster centroids 생성
#───────────────────────────────────────────────────────────────────────────────#

export CUDA_VISIBLE_DEVICES=0

# Mode 선택: "generate" 또는 "fit"
MODE="generate"

# Model paths
CKPT_PATH="runwayml/stable-diffusion-v1-5"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# Input options
PROMPTS_FILE="harmful_prompts.txt"  # harmful prompt 목록 파일
# LATENTS_PATH="collected_harmful_latents.pt"  # fit 모드 시 사용

# Generation options
NUM_SAMPLES_PER_PROMPT=10
CAPTURE_TIMESTEP=500
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5
HARMFUL_CLASS=2
CONFIDENCE_THRESHOLD=0.3

# Clustering options
N_CLUSTERS=10
POOLING="mean"

# Output
OUTPUT_PATH="cluster_centroids/harmful_clusters.pt"

# 실행
python mode_aware_guidance/cluster_harmful_latents.py \
    --mode $MODE \
    --ckpt_path "$CKPT_PATH" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --prompts_file "$PROMPTS_FILE" \
    --num_samples_per_prompt $NUM_SAMPLES_PER_PROMPT \
    --capture_timestep $CAPTURE_TIMESTEP \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --harmful_class $HARMFUL_CLASS \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --n_clusters $N_CLUSTERS \
    --pooling $POOLING \
    --output_path "$OUTPUT_PATH" \
    --save_latents

echo "========================================"
echo "Clustering complete!"
echo "Output: $OUTPUT_PATH"
echo "========================================"
