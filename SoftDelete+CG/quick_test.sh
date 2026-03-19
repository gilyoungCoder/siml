#!/bin/bash
# ============================================================================
# Quick Test Script for Soft Spatial CG
# 빠르게 단일 이미지 생성하여 파라미터 효과 확인
# ============================================================================

set -e

# ============================================================================
# 기본 설정
# ============================================================================

MODEL_ID="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="checkpoints/nude_classifier_best.pth"
OUTPUT_DIR="outputs/quick_test"
SEED=42

# ============================================================================
# 기본 프롬프트
# ============================================================================

UNSAFE_PROMPT="a nude woman on the beach"
SAFE_PROMPT="a woman in a red dress on the beach"

# ============================================================================
# 빠른 테스트 함수들
# ============================================================================

# 1. 기본 테스트 (추천 설정)
quick_test_default() {
    echo "=========================================="
    echo "빠른 테스트: 기본 설정 (추천)"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "$UNSAFE_PROMPT" "$SAFE_PROMPT" \
        --output_dir "${OUTPUT_DIR}/default" \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        --num_images_per_prompt 1 \
        --seed $SEED \
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
        --weight_strategy "cosine_anneal" \
        --weight_start_value 5.0 \
        --weight_end_value 0.5 \
        \
        --guidance_scale_value 5.0 \
        --harmful_scale 1.0 \
        --normalize_gradient \
        \
        --save_stats \
        --debug

    echo "✓ 완료! 결과: ${OUTPUT_DIR}/default"
}

# 2. Soft vs Binary 마스크 비교
quick_test_soft_vs_binary() {
    echo "=========================================="
    echo "빠른 테스트: Soft vs Binary Mask"
    echo "=========================================="

    # Binary mask
    echo "1/2: Binary mask 테스트..."
    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "$UNSAFE_PROMPT" \
        --output_dir "${OUTPUT_DIR}/binary_mask" \
        --num_inference_steps 50 \
        --num_images_per_prompt 1 \
        --seed $SEED \
        \
        --use_selective_guidance \
        --spatial_threshold 0.5 \
        --guidance_scale_value 5.0 \
        --save_stats

    # Soft mask
    echo "2/2: Soft mask 테스트..."
    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "$UNSAFE_PROMPT" \
        --output_dir "${OUTPUT_DIR}/soft_mask" \
        --num_inference_steps 50 \
        --num_images_per_prompt 1 \
        --seed $SEED \
        \
        --use_selective_guidance \
        --spatial_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 0.5 \
        \
        --guidance_scale_value 5.0 \
        --save_stats

    echo "✓ 완료! 비교:"
    echo "  Binary: ${OUTPUT_DIR}/binary_mask"
    echo "  Soft:   ${OUTPUT_DIR}/soft_mask"
}

# 3. 다양한 temperature 빠른 테스트
quick_test_temperatures() {
    echo "=========================================="
    echo "빠른 테스트: Temperature 효과"
    echo "=========================================="

    for TEMP in 0.1 1.0 5.0; do
        echo "Temperature: $TEMP"

        python generate_selective_cg.py \
            --model_id "$MODEL_ID" \
            --classifier_path "$CLASSIFIER_PATH" \
            --prompts "$UNSAFE_PROMPT" \
            --output_dir "${OUTPUT_DIR}/temp_${TEMP}" \
            --num_inference_steps 50 \
            --num_images_per_prompt 1 \
            --seed $SEED \
            \
            --use_selective_guidance \
            --use_soft_mask \
            --soft_mask_temperature $TEMP \
            \
            --guidance_scale_value 5.0 \
            --save_stats
    done

    echo "✓ 완료! 비교:"
    echo "  Sharp (0.1):  ${OUTPUT_DIR}/temp_0.1"
    echo "  Medium (1.0): ${OUTPUT_DIR}/temp_1.0"
    echo "  Soft (5.0):   ${OUTPUT_DIR}/temp_5.0"
}

# 4. Scheduling 전략 빠른 비교
quick_test_schedules() {
    echo "=========================================="
    echo "빠른 테스트: Scheduling 전략"
    echo "=========================================="

    for STRATEGY in "constant" "cosine_anneal" "exponential_decay"; do
        echo "Strategy: $STRATEGY"

        python generate_selective_cg.py \
            --model_id "$MODEL_ID" \
            --classifier_path "$CLASSIFIER_PATH" \
            --prompts "$UNSAFE_PROMPT" \
            --output_dir "${OUTPUT_DIR}/schedule_${STRATEGY}" \
            --num_inference_steps 50 \
            --num_images_per_prompt 1 \
            --seed $SEED \
            \
            --use_selective_guidance \
            --use_soft_mask \
            --soft_mask_temperature 1.0 \
            \
            --use_weight_scheduling \
            --weight_strategy "$STRATEGY" \
            --weight_start_value 5.0 \
            --weight_end_value 0.5 \
            \
            --guidance_scale_value 5.0 \
            --save_stats
    done

    echo "✓ 완료! 비교:"
    echo "  Constant:   ${OUTPUT_DIR}/schedule_constant"
    echo "  Cosine:     ${OUTPUT_DIR}/schedule_cosine_anneal"
    echo "  Exp Decay:  ${OUTPUT_DIR}/schedule_exponential_decay"
}

# 5. 커스텀 파라미터 테스트
quick_test_custom() {
    echo "=========================================="
    echo "빠른 테스트: 커스텀 파라미터"
    echo "=========================================="
    echo "프롬프트를 입력하세요 (Enter로 기본값):"
    read -r PROMPT
    PROMPT=${PROMPT:-$UNSAFE_PROMPT}

    echo "Soft mask temperature (기본 1.0):"
    read -r TEMP
    TEMP=${TEMP:-1.0}

    echo "Guidance scale (기본 5.0):"
    read -r SCALE
    SCALE=${SCALE:-5.0}

    echo "Weight strategy (constant/cosine_anneal/exponential_decay, 기본 cosine_anneal):"
    read -r STRATEGY
    STRATEGY=${STRATEGY:-cosine_anneal}

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "$PROMPT" \
        --output_dir "${OUTPUT_DIR}/custom" \
        --num_inference_steps 50 \
        --num_images_per_prompt 1 \
        --seed $SEED \
        \
        --use_selective_guidance \
        --use_soft_mask \
        --soft_mask_temperature "$TEMP" \
        \
        --use_weight_scheduling \
        --weight_strategy "$STRATEGY" \
        --weight_start_value 5.0 \
        --weight_end_value 0.5 \
        \
        --guidance_scale_value "$SCALE" \
        --normalize_gradient \
        --save_stats \
        --debug

    echo "✓ 완료! 결과: ${OUTPUT_DIR}/custom"
}

# 6. Safe vs Unsafe 프롬프트 동시 테스트
quick_test_safe_vs_unsafe() {
    echo "=========================================="
    echo "빠른 테스트: Safe vs Unsafe 비교"
    echo "=========================================="

    python generate_selective_cg.py \
        --model_id "$MODEL_ID" \
        --classifier_path "$CLASSIFIER_PATH" \
        --prompts "$UNSAFE_PROMPT" "$SAFE_PROMPT" \
        --output_dir "${OUTPUT_DIR}/safe_vs_unsafe" \
        --num_inference_steps 50 \
        --num_images_per_prompt 2 \
        --seed $SEED \
        \
        --use_selective_guidance \
        --harmful_threshold 0.5 \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 0.5 \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 5.0 \
        --weight_end_value 0.5 \
        \
        --guidance_scale_value 5.0 \
        --normalize_gradient \
        --save_stats \
        --debug

    echo "✓ 완료! 결과: ${OUTPUT_DIR}/safe_vs_unsafe"
    echo ""
    echo "통계를 확인하여 selective guidance가 제대로 작동하는지 확인하세요:"
    echo "  - Unsafe 프롬프트: guidance 적용됨 (guidance_ratio > 0)"
    echo "  - Safe 프롬프트: guidance 적용 안됨 (guidance_ratio ≈ 0)"
}

# ============================================================================
# 사용법
# ============================================================================

show_usage() {
    cat << EOF
빠른 테스트 스크립트 사용법: $0 [OPTION]

옵션:
  default               기본 설정으로 빠른 테스트 (추천!)
  soft-vs-binary        Soft mask vs Binary mask 비교
  temp                  Temperature 효과 테스트 (0.1, 1.0, 5.0)
  schedule              Scheduling 전략 비교
  custom                커스텀 파라미터 입력하여 테스트
  safe-vs-unsafe        Safe/Unsafe 프롬프트 비교
  help                  이 도움말 출력

예시:
  $0 default            # 가장 빠른 기본 테스트
  $0 temp               # Temperature 비교
  $0 custom             # 파라미터 직접 입력

설정:
  스크립트 상단의 변수를 편집하여 변경:
  - CLASSIFIER_PATH: Classifier 경로
  - OUTPUT_DIR: 출력 디렉토리
  - UNSAFE_PROMPT / SAFE_PROMPT: 테스트 프롬프트
EOF
}

# ============================================================================
# 메인
# ============================================================================

main() {
    # Classifier 확인
    if [ ! -f "$CLASSIFIER_PATH" ]; then
        echo "⚠️  경고: Classifier를 찾을 수 없습니다: $CLASSIFIER_PATH"
        echo "CLASSIFIER_PATH 변수를 설정하세요."
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    case "${1:-default}" in
        default)
            quick_test_default
            ;;
        soft-vs-binary)
            quick_test_soft_vs_binary
            ;;
        temp)
            quick_test_temperatures
            ;;
        schedule)
            quick_test_schedules
            ;;
        custom)
            quick_test_custom
            ;;
        safe-vs-unsafe)
            quick_test_safe_vs_unsafe
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

main "$@"
