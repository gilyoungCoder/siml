#!/bin/bash
# ============================================================================
# Mode-Aware Classifier Guidance - Violence
# Clustering centroid 기반 mode-aware guidance를 사용한 violence 억제 이미지 생성
#
# 각 cluster(mode)별로 다른 guidance scale을 적용하여 유해 콘텐츠 억제
# ============================================================================

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:/mnt/home/yhgil99/unlearning/SoftDelete+CG"

set -e

# ============================================================================
# 경로 설정
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"

# Clustering centroid 경로 (multi-timestep 통합 파일)
CENTROIDS_PATH="./cluster_centroids/violence_clusters.pt"

OUTPUT_DIR="./scg_outputs/mode_aware_violence"

# ============================================================================
# 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NUM_SAMPLES_PER_PROMPT=1
SEED=42

# ============================================================================
# Mode-Aware Guidance 파라미터
# ============================================================================

# Base guidance scale (모든 cluster에 기본 적용)
BASE_GUIDANCE_SCALE=25.0

# Harmful class (0: benign, 1: safe, 2: harmful)
HARMFUL_CLASS=2

# Guidance 적용 구간
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# (Optional) Cluster별 scale 설정 파일
# 없으면 모든 cluster에 동일한 scale 적용
# MODE_SCALES_FILE="./cluster_centroids/violence_mode_scales.json"

# ============================================================================
# 프롬프트 파일
# ============================================================================

PROMPTS_FILE="./prompts/violence_50.txt"

# ============================================================================
# 실행
# ============================================================================

echo "========================================"
echo "Mode-Aware Classifier Guidance - Violence"
echo "========================================"
echo ""
echo "설정:"
echo "  Classifier: $CLASSIFIER_PATH"
echo "  Centroids:  $CENTROIDS_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "  Prompts:    $PROMPTS_FILE"
echo ""
echo "Guidance 파라미터:"
echo "  Base Scale:     $BASE_GUIDANCE_SCALE"
echo "  Harmful Class:  $HARMFUL_CLASS"
echo "  Active Steps:   $GUIDANCE_START_STEP ~ $GUIDANCE_END_STEP"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_DIR"

python generate_with_mode_aware_guidance.py \
    --ckpt_path "$CKPT_PATH" \
    --classifier_ckpt "$CLASSIFIER_PATH" \
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
    --seed $SEED \
    --save_metadata

echo ""
echo "========================================"
echo "Generation complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
