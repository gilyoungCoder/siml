#!/bin/bash
# ============================================================================
# Adaptive CG V2 - 진짜 Adaptive!
# 3가지 Adaptive 메커니즘:
#   1. Adaptive Threshold by Timestep
#   2. Adaptive Guidance Scale by Timestep
#   3. Spatial-Adaptive Guidance by GradCAM Score
# ============================================================================
export CUDA_VISIBLE_DEVICES=7

set -e

# ============================================================================
# 설정 변수
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"

# # nudity
# CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
# OUTPUT_BASE_DIR="scg_outputs/adaptive_v2/nudtiy_experiments"
# PROMPT_FILE="./prompts/sexual_50.txt"

# violence
PROMPT_FILE="./prompts/violence_50.txt"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
OUTPUT_BASE_DIR="scg_outputs/adaptive_v2/violence_experiments"

# 프롬프트 파일 경로

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=3
SEED=42

# ============================================================================
# 실험 1: Binary Mask Only (Baseline)
# ============================================================================

exp_binary_baseline() {
    echo "=========================================="
    echo "실험 1: Binary Mask Baseline"
    echo "  Threshold: 0.5 (고정)"
    echo "  Guidance Scale: 10.0 → 0.5 (Cosine Anneal)"
    echo "  Mask: Binary (0 or 1)"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/binary_baseline"

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
        --harmful_threshold 0.5 \
        --spatial_threshold 0.5 \
        --guidance_scale 5.0 \
        --harmful_scale 1.5 \
        --use_bidirectional \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 10.0 \
        --weight_end_value 0.5 \
        \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        --save_visualizations

    echo ""
    echo "✓ Binary Baseline 완료!"
}

# ============================================================================
# 실험 2: Adaptive Threshold
# ============================================================================

exp_adaptive_threshold() {
    echo "=========================================="
    echo "실험 2: Adaptive Threshold"
    echo "  Threshold: 0.7 → 0.3 (Cosine Anneal)"
    echo "  Guidance Scale: 10.0 → 0.5 (Cosine Anneal)"
    echo "  Mask: Binary"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/adaptive_threshold"

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
        --harmful_threshold 0.5 \
        --spatial_threshold 0.5 \
        --guidance_scale 5.0 \
        --harmful_scale 1.5 \
        --use_bidirectional \
        \
        --use_adaptive_threshold \
        --threshold_strategy "cosine_anneal" \
        --threshold_start_value 0.7 \
        --threshold_end_value 0.3 \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 10.0 \
        --weight_end_value 0.5 \
        \
        --normalize_gradient \
        --save_visualizations

    echo ""
    echo "✓ Adaptive Threshold 완료!"
}

# ============================================================================
# 실험 3: Heatmap-Weighted Guidance
# ============================================================================

exp_heatmap_weighted() {
    echo "=========================================="
    echo "실험 3: Heatmap-Weighted Guidance"
    echo "  Threshold: 0.5 (고정)"
    echo "  Guidance Scale: 10.0 → 0.5 (Cosine Anneal)"
    echo "  Mask: Binary * Heatmap (Pixel-wise Adaptive!)"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/heatmap_weighted"

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
        --harmful_threshold 0.5 \
        --spatial_threshold 0.5 \
        --guidance_scale 5.0 \
        --harmful_scale 1.5 \
        --use_bidirectional \
        \
        --use_heatmap_weighted_guidance \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 10.0 \
        --weight_end_value 0.5 \
        \
        --normalize_gradient \
        --save_visualizations

    echo ""
    echo "✓ Heatmap-Weighted 완료!"
}

# ============================================================================
# 실험 4: Full Adaptive (모든 기능)
# ============================================================================

exp_full_adaptive() {
    echo "=========================================="
    echo "실험 4: Full Adaptive (All Features)"
    echo "  Threshold: 0.7 → 0.3 (Adaptive)"
    echo "  Guidance Scale: 10.0 → 0.5 (Adaptive)"
    echo "  Mask: Binary * Heatmap (Pixel-wise Adaptive)"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/full_adaptive"

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
        --harmful_threshold 0.5 \
        --spatial_threshold 0.5 \
        --guidance_scale 5.0 \
        --harmful_scale 1.5 \
        --use_bidirectional \
        \
        --use_adaptive_threshold \
        --threshold_strategy "cosine_anneal" \
        --threshold_start_value 0.7 \
        --threshold_end_value 0.3 \
        \
        --use_heatmap_weighted_guidance \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 10.0 \
        --weight_end_value 0.5 \
        \
        --normalize_gradient \
        --save_visualizations

    echo ""
    echo "✓ Full Adaptive 완료!"
}

# ============================================================================
# 메인 함수
# ============================================================================

run_all_experiments() {
    echo "============================================"
    echo "Adaptive CG V2 - 모든 실험 실행"
    echo "프롬프트 파일: $PROMPT_FILE"
    echo "============================================"

    exp_binary_baseline
    exp_adaptive_threshold
    exp_heatmap_weighted
    exp_full_adaptive

    echo ""
    echo "✓ 모든 실험 완료!"
    echo "결과: $OUTPUT_BASE_DIR"
}

# ============================================================================
# 사용법
# ============================================================================

show_usage() {
    cat << EOF
Adaptive CG V2 실험 스크립트

사용법: $0 [OPTION]

옵션:
  all               모든 실험 (1-4)

  binary            실험 1: Binary Baseline
  adaptive-threshold 실험 2: Adaptive Threshold
  heatmap           실험 3: Heatmap-Weighted
  full              실험 4: Full Adaptive

  help              도움말

예시:
  $0 all                  # 모든 실험
  $0 binary               # Binary baseline만
  $0 full                 # Full adaptive만

실험 비교:
  1. Binary Baseline: 고정 threshold, binary mask
  2. Adaptive Threshold: 시간에 따라 threshold 변화 (0.7→0.3)
  3. Heatmap-Weighted: GradCAM 값으로 pixel-wise guidance 강도 조절
  4. Full Adaptive: 모든 adaptive 기능 활성화

핵심 차별점:
  ✅ Sigmoid soft masking 제거 (명확한 binary)
  ✅ Adaptive threshold by timestep
  ✅ Adaptive guidance scale by timestep
  ✅ Pixel-wise adaptive guidance by GradCAM
EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    if [ ! -f "$CLASSIFIER_PATH" ]; then
        echo "⚠️  Classifier를 찾을 수 없습니다: $CLASSIFIER_PATH"
        exit 1
    fi

    if [ ! -f "$PROMPT_FILE" ]; then
        echo "⚠️  프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
        exit 1
    fi

    mkdir -p "$OUTPUT_BASE_DIR"

    case "${1:-help}" in
        all)
            run_all_experiments
            ;;
        binary)
            exp_binary_baseline
            ;;
        adaptive-threshold)
            exp_adaptive_threshold
            ;;
        heatmap)
            exp_heatmap_weighted
            ;;
        full)
            exp_full_adaptive
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "⚠️  알 수 없는 옵션: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
