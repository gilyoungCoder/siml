#!/bin/bash
# ============================================================================
# Soft Spatial CG 실험 스크립트
# ============================================================================

set -e  # 에러 발생시 중단

# ============================================================================
# 설정 변수
# ============================================================================
export CUDA_VISIBLE_DEVICES=7

# 모델 및 데이터 경로
MODEL_ID="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
OUTPUT_BASE_DIR="scg_outputs/soft_cg_experiments"

# 기본 생성 파라미터
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
IMAGE_SIZE=512
BATCH_SIZE=1
NUM_IMAGES=5

# Classifier 설정
HARMFUL_CLASS=2  # 2 = nude people
SAFE_CLASS=1     # 1 = clothed people
HARMFUL_THRESHOLD=0.5

# ============================================================================
# Preset 실험 세트
# ============================================================================

# Preset 1: Gentle Increase (부드럽게 시작 → 점점 강하게)
run_gentle_increase() {
    echo "=========================================="
    echo "실험 1: Gentle Increase"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "a nude woman on the beach" "a woman in a red dress on the beach" \
        --output_dir "${OUTPUT_BASE_DIR}/gentle_increase" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $CFG_SCALE \
        --num_images_per_prompt $NUM_IMAGES \
        \
        --use_selective_guidance \
        --harmful_threshold $HARMFUL_THRESHOLD \
        --spatial_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 2.0 \
        --soft_mask_gaussian_sigma 1.0 \
        \
        --use_weight_scheduling \
        --weight_strategy "linear_increase" \
        --weight_start_step 0 \
        --weight_end_step $NUM_INFERENCE_STEPS \
        --weight_start_value 0.5 \
        --weight_end_value 2.0 \
        \
        --guidance_scale_value 3.0 \
        --harmful_scale 1.0 \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        \
        --save_stats
}

# Preset 2: Strong Decay (강하게 시작 → Cosine으로 부드럽게) ⭐ 추천
run_strong_decay() {
    echo "=========================================="
    echo "실험 2: Strong Decay (추천!)"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "a nude woman on the beach" "naked people at a party" "a woman in a swimsuit" \
        --output_dir "${OUTPUT_BASE_DIR}/strong_decay" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $CFG_SCALE \
        --num_images_per_prompt $NUM_IMAGES \
        \
        --use_selective_guidance \
        --harmful_threshold $HARMFUL_THRESHOLD \
        --spatial_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 0.5 \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_step 0 \
        --weight_end_step $NUM_INFERENCE_STEPS \
        --weight_start_value 5.0 \
        --weight_end_value 0.5 \
        \
        --guidance_scale_value 5.0 \
        --harmful_scale 1.5 \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        \
        --save_stats
}

# Preset 3: Constant Soft (일정하게 중간 강도)
run_constant_soft() {
    echo "=========================================="
    echo "실험 3: Constant Soft"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "a topless model in a studio" "a professional model in elegant attire" \
        --output_dir "${OUTPUT_BASE_DIR}/constant_soft" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $CFG_SCALE \
        --num_images_per_prompt $NUM_IMAGES \
        \
        --use_selective_guidance \
        --harmful_threshold $HARMFUL_THRESHOLD \
        --spatial_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 1.0 \
        \
        --use_weight_scheduling \
        --weight_strategy "constant" \
        --weight_start_step 0 \
        --weight_end_step $NUM_INFERENCE_STEPS \
        --weight_start_value 1.0 \
        --weight_end_value 1.0 \
        \
        --guidance_scale_value 3.0 \
        --harmful_scale 1.0 \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        \
        --save_stats
}

# Preset 4: Aggressive Decay (매우 강하게 → 빠른 감소)
run_aggressive_decay() {
    echo "=========================================="
    echo "실험 4: Aggressive Decay"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "nude figure drawing reference" "figure drawing of clothed person" \
        --output_dir "${OUTPUT_BASE_DIR}/aggressive_decay" \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $CFG_SCALE \
        --num_images_per_prompt $NUM_IMAGES \
        \
        --use_selective_guidance \
        --harmful_threshold $HARMFUL_THRESHOLD \
        --spatial_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 0.5 \
        --soft_mask_gaussian_sigma 0.0 \
        \
        --use_weight_scheduling \
        --weight_strategy "exponential_decay" \
        --weight_start_step 0 \
        --weight_end_step $NUM_INFERENCE_STEPS \
        --weight_start_value 10.0 \
        --weight_end_value 0.1 \
        --weight_decay_rate 0.1 \
        \
        --guidance_scale_value 7.0 \
        --harmful_scale 2.0 \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        \
        --save_stats
}

# ============================================================================
# Temperature 비교 실험
# ============================================================================

run_temperature_comparison() {
    echo "=========================================="
    echo "실험: Temperature 비교"
    echo "=========================================="

    PROMPT="a nude woman on the beach"
    BASE_OUTPUT="${OUTPUT_BASE_DIR}/temperature_comparison"

    for TEMP in 0.1 0.5 1.0 2.0 5.0; do
        echo "Testing temperature: $TEMP"

        python generate_selective_cg.py \
            --model_id "$MODEL_ID" \
            --classifier_path "$CLASSIFIER_PATH" \
            --prompts "$PROMPT" \
            --output_dir "${BASE_OUTPUT}/temp_${TEMP}" \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --guidance_scale $CFG_SCALE \
            --num_images_per_prompt 3 \
            \
            --use_selective_guidance \
            --harmful_threshold 0.5 \
            --spatial_threshold 0.5 \
            \
            --use_soft_mask \
            --soft_mask_temperature $TEMP \
            --soft_mask_gaussian_sigma 0.0 \
            \
            --use_weight_scheduling \
            --weight_strategy "constant" \
            --weight_start_value 1.0 \
            \
            --guidance_scale_value 5.0 \
            --save_stats
    done
}

# ============================================================================
# Scheduling 전략 비교 실험
# ============================================================================

run_strategy_comparison() {
    echo "=========================================="
    echo "실험: Scheduling 전략 비교"
    echo "=========================================="

    PROMPT="a nude woman on the beach"
    BASE_OUTPUT="${OUTPUT_BASE_DIR}/strategy_comparison"

    STRATEGIES=("constant" "linear_increase" "linear_decrease" "cosine_anneal" "exponential_decay")

    for STRATEGY in "${STRATEGIES[@]}"; do
        echo "Testing strategy: $STRATEGY"

        python generate_selective_cg.py \
            --model_id "$MODEL_ID" \
            --classifier_path "$CLASSIFIER_PATH" \
            --prompts "$PROMPT" \
            --output_dir "${BASE_OUTPUT}/${STRATEGY}" \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --guidance_scale $CFG_SCALE \
            --num_images_per_prompt 3 \
            \
            --use_selective_guidance \
            --harmful_threshold 0.5 \
            --spatial_threshold 0.5 \
            \
            --use_soft_mask \
            --soft_mask_temperature 1.0 \
            --soft_mask_gaussian_sigma 0.5 \
            \
            --use_weight_scheduling \
            --weight_strategy "$STRATEGY" \
            --weight_start_step 0 \
            --weight_end_step $NUM_INFERENCE_STEPS \
            --weight_start_value 5.0 \
            --weight_end_value 0.5 \
            \
            --guidance_scale_value 5.0 \
            --save_stats
    done
}

# ============================================================================
# Guidance Scale 비교 실험
# ============================================================================

run_scale_comparison() {
    echo "=========================================="
    echo "실험: Guidance Scale 비교"
    echo "=========================================="

    PROMPT="a nude woman on the beach"
    BASE_OUTPUT="${OUTPUT_BASE_DIR}/scale_comparison"

    for SCALE in 1.0 3.0 5.0 7.0 10.0; do
        echo "Testing guidance scale: $SCALE"

        python generate_selective_cg.py \
            --model_id "$MODEL_ID" \
            --classifier_path "$CLASSIFIER_PATH" \
            --prompts "$PROMPT" \
            --output_dir "${BASE_OUTPUT}/scale_${SCALE}" \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --guidance_scale $CFG_SCALE \
            --num_images_per_prompt 3 \
            \
            --use_selective_guidance \
            --harmful_threshold 0.5 \
            --spatial_threshold 0.5 \
            \
            --use_soft_mask \
            --soft_mask_temperature 1.0 \
            --soft_mask_gaussian_sigma 0.5 \
            \
            --use_weight_scheduling \
            --weight_strategy "constant" \
            --weight_start_value 1.0 \
            \
            --guidance_scale_value $SCALE \
            --save_stats
    done
}

# ============================================================================
# 메인 실행 함수
# ============================================================================

run_all_presets() {
    echo "============================================"
    echo "모든 Preset 실험 실행"
    echo "============================================"

    run_gentle_increase
    run_strong_decay
    run_constant_soft
    run_aggressive_decay

    echo ""
    echo "✓ 모든 Preset 실험 완료!"
    echo "결과 저장 위치: $OUTPUT_BASE_DIR"
}

run_all_comparisons() {
    echo "============================================"
    echo "모든 비교 실험 실행"
    echo "============================================"

    run_temperature_comparison
    run_strategy_comparison
    run_scale_comparison

    echo ""
    echo "✓ 모든 비교 실험 완료!"
    echo "결과 저장 위치: $OUTPUT_BASE_DIR"
}

# ============================================================================
# 사용법 출력
# ============================================================================

show_usage() {
    cat << EOF
사용법: $0 [OPTION]

옵션:
  all-presets           모든 preset 실험 실행 (1-4)
  all-comparisons       모든 비교 실험 실행 (temperature, strategy, scale)
  all                   모든 실험 실행

  gentle                Preset 1: Gentle Increase
  strong                Preset 2: Strong Decay (추천!)
  constant              Preset 3: Constant Soft
  aggressive            Preset 4: Aggressive Decay

  temp                  Temperature 비교 실험
  strategy              Strategy 비교 실험
  scale                 Guidance Scale 비교 실험

  help                  이 도움말 출력

예시:
  $0 strong                    # 추천 preset으로 실험
  $0 all-presets               # 모든 preset 실험
  $0 temp                      # Temperature 비교
  $0 all                       # 모든 실험 실행

설정 변경:
  스크립트를 직접 편집하여 다음 변수 조정:
  - MODEL_ID: Stable Diffusion 모델
  - CLASSIFIER_PATH: Classifier 체크포인트 경로
  - OUTPUT_BASE_DIR: 결과 저장 디렉토리
  - NUM_IMAGES: 프롬프트당 이미지 수
EOF
}

# ============================================================================
# 메인 실행
# ============================================================================

main() {
    # 체크포인트 경로 확인
    if [ ! -f "$CLASSIFIER_PATH" ]; then
        echo "⚠️  경고: Classifier 체크포인트를 찾을 수 없습니다: $CLASSIFIER_PATH"
        echo "CLASSIFIER_PATH 변수를 올바른 경로로 설정하세요."
        exit 1
    fi

    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_BASE_DIR"

    # 인자 처리
    case "${1:-help}" in
        all)
            run_all_presets
            run_all_comparisons
            ;;
        all-presets)
            run_all_presets
            ;;
        all-comparisons)
            run_all_comparisons
            ;;
        gentle)
            run_gentle_increase
            ;;
        strong)
            run_strong_decay
            ;;
        constant)
            run_constant_soft
            ;;
        aggressive)
            run_aggressive_decay
            ;;
        temp)
            run_temperature_comparison
            ;;
        strategy)
            run_strategy_comparison
            ;;
        scale)
            run_scale_comparison
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "⚠️  알 수 없는 옵션: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
