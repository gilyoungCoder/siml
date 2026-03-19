#!/bin/bash
# ============================================================================
# Grid Search for Violence Classifier - Full Adaptive CG
# 각 파라미터 조합별로 이미지 생성 및 비교
# ============================================================================
export CUDA_VISIBLE_DEVICES=6

set -e

# ============================================================================
# 기본 설정
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
PROMPT_FILE="./prompts/violence_50.txt"
BASE_OUTPUT_DIR="scg_outputs/grid_search_violence"

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=123

# ============================================================================
# Grid Search 파라미터 정의
# ============================================================================

# Guidance Scale 옵션
GUIDANCE_SCALES=(7.0 8.0 10.0 12.0)

# Harmful Scale 옵션
HARMFUL_SCALES=(1.0 1.25 1.5)

# Spatial Threshold 옵션 (고정 threshold)
SPATIAL_THRESHOLDS=(0.2 0.3 0.4 0.5)

# Weight Scheduling 옵션 (start_value end_value)
WEIGHT_SCHEDULES=(
    "3.0 0.5"
    "4.0 1.0"
    "5.0 1.0"
    "8.0 1.0"
)

# Adaptive Threshold 옵션 (start_value end_value)
THRESHOLD_SCHEDULES=(
    "0.0 -1.5"
    "0.0 -2.0"
    "-0.5 -2.5"
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
echo "🔍 Grid Search for Violence - Starting"
echo "============================================"
echo "Base Output: $BASE_OUTPUT_DIR"
echo "Prompts: $PROMPT_FILE"
echo "Samples per prompt: $NSAMPLES"
echo ""
echo "Grid dimensions:"
echo "  Guidance Scales: ${GUIDANCE_SCALES[@]}"
echo "  Harmful Scales: ${HARMFUL_SCALES[@]}"
echo "  Spatial Thresholds: ${SPATIAL_THRESHOLDS[@]}"
echo "  Weight Schedules: ${#WEIGHT_SCHEDULES[@]} options"
echo "  Threshold Schedules: ${#THRESHOLD_SCHEDULES[@]} options"
echo ""

TOTAL_EXPERIMENTS=$((${#GUIDANCE_SCALES[@]} * ${#HARMFUL_SCALES[@]} * ${#SPATIAL_THRESHOLDS[@]} * ${#WEIGHT_SCHEDULES[@]} * ${#THRESHOLD_SCHEDULES[@]}))
echo "Total experiments: $TOTAL_EXPERIMENTS"
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
                    echo "Experiment [$EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS]: $EXPERIMENT_NAME"
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
                    echo "✅ Experiment $EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS complete!"
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
echo "🎉 Grid Search Complete!"
echo "============================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
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
