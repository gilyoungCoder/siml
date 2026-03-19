#!/usr/bin/env bash
#───────────────────────────────────────────────────────────────────────────────#
# Mode-Aware Classifier Guidance Image Generation
# 각 cluster(mode)별로 다른 guidance scale을 적용하여 이미지 생성
#───────────────────────────────────────────────────────────────────────────────#

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/mnt/home/yhgil99/unlearning/SoftDelete+CG"

# Model paths
CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
CENTROIDS_PATH="cluster_centroids/violence_clusters.pt"

# Input prompts
PROMPTS_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/violence/harm.txt"

# Generation options
NUM_SAMPLES_PER_PROMPT=1
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5

# Mode-aware guidance options
BASE_GUIDANCE_SCALE=100.0
HARMFUL_CLASS=2
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Optional: per-cluster scales (JSON file)
# MODE_SCALES_FILE="mode_scales.json"

# Output
OUTPUT_DIR="outputs/mode_aware_violence"

# 실행
nohup python generate_with_mode_aware_guidance.py \
    --ckpt_path "$CKPT_PATH" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --centroids_path "$CENTROIDS_PATH" \
    --prompts_file "$PROMPTS_FILE" \
    --num_samples_per_prompt $NUM_SAMPLES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --cfg_scale $CFG_SCALE \
    --base_guidance_scale $BASE_GUIDANCE_SCALE \
    --harmful_class $HARMFUL_CLASS \
    --guidance_start_step $GUIDANCE_START_STEP \
    --guidance_end_step $GUIDANCE_END_STEP \
    --output_dir "$OUTPUT_DIR" \
    --save_metadata \
    > generate_mode_aware.log 2>&1 &

echo "========================================"
echo "Mode-Aware Generation launched!"
echo "Output: $OUTPUT_DIR"
echo "Log: generate_mode_aware.log"
echo ""
echo "Monitor with: tail -f generate_mode_aware.log"
echo "========================================"
