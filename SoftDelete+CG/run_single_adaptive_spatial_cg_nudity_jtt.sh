#!/bin/bash
# ============================================================================
# Single Experiment: Adaptive Spatial CG - Nudity with JTT Stage2 Classifier
# JTT stage2 step_18400 classifier를 사용한 이미지 생성
# Usage: bash run_single_adaptive_spatial_cg_nudity_jtt.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:/mnt/home/yhgil99/unlearning/SoftDelete+CG"

set -e

# ============================================================================
# 경로 설정
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
# JTT Stage2 step_18400 classifier
CLASSIFIER_PATH="./work_dirs/jtt_stage2_T5800/checkpoints/step_18400/classifier.pth"
OUTPUT_DIR="./scg_outputs/nudity_jtt_step18400"

# ============================================================================
# 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=42

# Guidance 파라미터
GUIDANCE_SCALE=10.0
HARMFUL_SCALE=1.5
SPATIAL_THRESHOLD_START=0.4
SPATIAL_THRESHOLD_END=0.1
THRESHOLD_STRATEGY="linear_decrease"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# 프롬프트 파일
# ============================================================================

PROMPT_FILE="./prompts/sexual_50.txt"

# ============================================================================
# 실행
# ============================================================================

echo "========================================"
echo "Adaptive Spatial CG - Nudity (JTT step_18400)"
echo "========================================"
echo "Classifier: $CLASSIFIER_PATH"
echo "Output: $OUTPUT_DIR"
echo "Prompt file: $PROMPT_FILE"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

python generate_always_adaptive_spatial_cg.py \
    "$CKPT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --nsamples $NSAMPLES \
    --cfg_scale $CFG_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --seed $SEED \
    --classifier_ckpt "$CLASSIFIER_PATH" \
    --guidance_scale $GUIDANCE_SCALE \
    --spatial_threshold_start $SPATIAL_THRESHOLD_START \
    --spatial_threshold_end $SPATIAL_THRESHOLD_END \
    --threshold_strategy $THRESHOLD_STRATEGY \
    --guidance_start_step $GUIDANCE_START_STEP \
    --guidance_end_step $GUIDANCE_END_STEP \
    --harmful_scale $HARMFUL_SCALE \
    --use_bidirectional \
    --save_visualizations

echo "========================================"
echo "Generation complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
