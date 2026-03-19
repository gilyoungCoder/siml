#!/bin/bash
# ============================================================================
# Grid Search for Violence Classifier - EXTRA Experiments
# 기존 실험과 다른 파라미터 범위로 추가 실험
# ============================================================================
export CUDA_VISIBLE_DEVICES=6

set -e

# ============================================================================
# 기본 설정
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
PROMPT_FILE="./prompts/violence_50.txt"
BASE_OUTPUT_DIR="scg_outputs/grid_search_violence_extra"  # 다른 폴더!

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=123

# ============================================================================
# Grid Search 파라미터 정의 - EXTRA Experiments
# ============================================================================

# 더 강한 Guidance Scale 테스트
GUIDANCE_SCALES=(15.0 18.0 20.0)

# Harmful Scale - 더 극단적인 값들
HARMFUL_SCALES=(0.5 0.75 2.0)

# Spatial Threshold - 더 극단적인 값들
SPATIAL_THRESHOLDS=(0.1 0.15 0.6)

# Weight Scheduling - 강한 초반, 약한 후반
WEIGHT_SCHEDULES=(
    "6.0 0.5"
    "8.0 0.5"
    "10.0 0.5"
)

# Adaptive Threshold - 더 극단적인 범위
THRESHOLD_SCHEDULES=(
    "1.0 -1.0"
    "-1.0 -3.0"
)

# ============================================================================
# Classifier 확인
# ============================================================================

if [ ! -f "$CLASSIFIER_PATH" ]; then
    echo "❌ Error: Classifier not found at $CLASSIFIER_PATH"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompt file not found at $PROMPT_FILE"
    exit 1
fi

# ============================================================================
# Grid Search 실행
# ============================================================================

echo "============================================"
echo "🔍 Grid Search for Violence - EXTRA Experiments"
echo "============================================"
echo "Base Output: $BASE_OUTPUT_DIR"
echo "Prompts: $PROMPT_FILE"
echo "Samples per prompt: $NSAMPLES"
echo ""
echo "Grid dimensions (EXTRA range):"
echo "  Guidance Scales: ${GUIDANCE_SCALES[@]} (high values)"
echo "  Harmful Scales: ${HARMFUL_SCALES[@]} (extreme values)"
echo "  Spatial Thresholds: ${SPATIAL_THRESHOLDS[@]} (extreme values)"
echo "  Weight Schedules: ${#WEIGHT_SCHEDULES[@]} options (strong start)"
echo "  Threshold Schedules: ${#THRESHOLD_SCHEDULES[@]} options (extreme range)"
echo ""

TOTAL_EXPERIMENTS=$((${#GUIDANCE_SCALES[@]} * ${#HARMFUL_SCALES[@]} * ${#SPATIAL_THRESHOLDS[@]} * ${#WEIGHT_SCHEDULES[@]} * ${#THRESHOLD_SCHEDULES[@]}))
echo "Total EXTRA experiments: $TOTAL_EXPERIMENTS"
echo "============================================"
echo ""

EXPERIMENT_COUNT=0

# Loop through all combinations
for guidance_scale in "${GUIDANCE_SCALES[@]}"; do
    for harmful_scale in "${HARMFUL_SCALES[@]}"; do
        for spatial_threshold in "${SPATIAL_THRESHOLDS[@]}"; do
            for weight_schedule in "${WEIGHT_SCHEDULES[@]}"; do
                read weight_start weight_end <<< "$weight_schedule"

                for threshold_schedule in "${THRESHOLD_SCHEDULES[@]}"; do
                    read threshold_start threshold_end <<< "$threshold_schedule"

                    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))

                    # 폴더 이름 생성 (파라미터 포함)
                    EXPERIMENT_NAME="gs${guidance_scale}_hs${harmful_scale}_st${spatial_threshold}_ws${weight_start}-${weight_end}_ts${threshold_start}-${threshold_end}"
                    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

                    echo ""
                    echo "========================================"
                    echo "EXTRA Experiment [$EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS]: $EXPERIMENT_NAME"
                    echo "========================================"
                    echo "  Guidance Scale: $guidance_scale"
                    echo "  Harmful Scale: $harmful_scale"
                    echo "  Spatial Threshold: $spatial_threshold"
                    echo "  Weight Schedule: $weight_start → $weight_end"
                    echo "  Threshold Schedule: $threshold_start → $threshold_end"
                    echo "  Output: $OUTPUT_DIR"
                    echo "========================================"
                    echo ""

                    # 실험 실행
                    python generate_selective_cg.py \
                        "$CKPT_PATH" \
                        --prompt_file "$PROMPT_FILE" \
                        --output_dir "$OUTPUT_DIR" \
                        --nsamples $NSAMPLES \
                        --num_inference_steps $NUM_INFERENCE_STEPS \
                        --cfg_scale $CFG_SCALE \
                        --seed $SEED \
                        \
                        --selective_guidance \
                        --classifier_ckpt "$CLASSIFIER_PATH" \
                        --harmful_threshold -0.5 \
                        --spatial_threshold $spatial_threshold \
                        --guidance_scale $guidance_scale \
                        --harmful_scale $harmful_scale \
                        --use_bidirectional \
                        \
                        --use_adaptive_threshold \
                        --threshold_strategy "cosine_anneal" \
                        --threshold_start_value $threshold_start \
                        --threshold_end_value $threshold_end \
                        \
                        --use_heatmap_weighted_guidance \
                        \
                        --use_weight_scheduling \
                        --weight_strategy "cosine_anneal" \
                        --weight_start_value $weight_start \
                        --weight_end_value $weight_end \
                        \
                        --normalize_gradient \
                        --gradient_norm_type "l2" \
                        --save_visualizations

                    echo ""
                    echo "✅ EXTRA Experiment $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS complete!"
                    echo "   Output: $OUTPUT_DIR"
                    echo ""

                done
            done
        done
    done
done

# ============================================================================
# 결과 요약
# ============================================================================

echo ""
echo "============================================"
echo "🎉 EXTRA Grid Search Complete!"
echo "============================================"
echo "Total EXTRA experiments: $TOTAL_EXPERIMENTS"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo ""
echo "Experiment folders:"
ls -1 "$BASE_OUTPUT_DIR" | head -20
if [ $(ls -1 "$BASE_OUTPUT_DIR" | wc -l) -gt 20 ]; then
    echo "... and $(($(ls -1 "$BASE_OUTPUT_DIR" | wc -l) - 20)) more"
fi
echo ""
echo "To compare results, check each folder:"
echo "  $BASE_OUTPUT_DIR/gs*"
echo "============================================"
