#!/bin/bash
# ============================================================================
# Adaptive Soft Spatial CG 실험 스크립트
# generate_selective_cg.py의 새로운 기능 사용
# ============================================================================
export CUDA_VISIBLE_DEVICES=7

set -e

# ============================================================================
# 설정 변수
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"

# PROMPT_FILE="./prompts/sexual_50.txt"
# CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
# OUTPUT_BASE_DIR="scg_outputs/adaptive_soft_cg/nudity_experiments"

# 프롬프트 파일 경로
PROMPT_FILE="./prompts/violence_50.txt"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
OUTPUT_BASE_DIR="scg_outputs/adaptive_soft_cg/violence_experiments"

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=3
SEED=42

# ============================================================================
# Helper Functions
# ============================================================================

create_temp_prompt_file() {
    local TEMP_FILE=$(mktemp)
    shift
    for prompt in "$@"; do
        echo "$prompt" >> "$TEMP_FILE"
    done
    echo "$TEMP_FILE"
}

# ============================================================================
# 결과 분석 함수
# ============================================================================

analyze_results() {
    local OUTPUT_DIR="$1"

    echo ""
    echo "============================================"
    echo "📊 배치 생성 결과 분석"
    echo "============================================"

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "⚠️  결과 디렉토리를 찾을 수 없습니다: $OUTPUT_DIR"
        return 1
    fi

    python << EOF
import os
import json
from pathlib import Path
from collections import defaultdict

output_dir = Path("$OUTPUT_DIR")

# 생성된 이미지 수집
image_files = list(output_dir.glob("*.png"))
print(f"\n총 생성된 이미지: {len(image_files)}개\n")

# 프롬프트별 분석
prompts_analyzed = set()
harmful_detected = []
safe_detected = []

for img_file in sorted(image_files):
    # 파일명에서 프롬프트 추출 (예: 0000_00_prompt_text.png)
    parts = img_file.stem.split('_', 2)
    if len(parts) >= 3:
        prompt_text = parts[2].replace('_', ' ')

        if prompt_text not in prompts_analyzed:
            prompts_analyzed.add(prompt_text)

            # 파일명에 harmful 관련 키워드가 있는지 확인
            # (실제로는 generate_selective_cg.py의 출력 로그를 파싱해야 함)
            prompt_lower = prompt_text.lower()

            # 단순 휴리스틱 (실제 분류는 classifier 출력 확인 필요)
            if any(word in prompt_lower for word in ['nude', 'naked', 'topless', 'licking', 'stabbed', 'blood']):
                harmful_detected.append(prompt_text[:60])
            else:
                safe_detected.append(prompt_text[:60])

print("="*60)
print(f"Harmful 감지된 프롬프트: {len(harmful_detected)}개")
print("="*60)
for i, p in enumerate(harmful_detected[:10], 1):
    print(f"  {i}. {p}...")

print()
print("="*60)
print(f"Safe로 분류된 프롬프트: {len(safe_detected)}개")
print("="*60)
for i, p in enumerate(safe_detected[:10], 1):
    print(f"  {i}. {p}...")

print()
print("="*60)
print("시각화 파일 확인")
print("="*60)

vis_dir = output_dir / "visualizations"
if vis_dir.exists():
    vis_files = list(vis_dir.glob("*.png"))
    print(f"  시각화 파일: {len(vis_files)}개")

    # 분석 파일 카운트
    analysis_files = [f for f in vis_files if 'analysis' in f.name]
    heatmap_files = [f for f in vis_files if 'heatmap' in f.name]

    print(f"  - Analysis 파일: {len(analysis_files)}개")
    print(f"  - Heatmap 파일: {len(heatmap_files)}개")
else:
    print("  ⚠️  visualizations 디렉토리가 없습니다")
    print("     --save_visualizations 옵션을 사용하세요")

print()
print("="*60)
print("✓ 분석 완료!")
print("="*60)
print(f"결과 위치: {output_dir}")
EOF
}

# ============================================================================
# 핵심 실험: Soft Masking + Weight Scheduling 조합
# ============================================================================

# 실험 1: Gentle Start → Strong End (점점 강하게)
exp_gentle_to_strong() {
    echo "=========================================="
    echo "실험 1: Gentle → Strong"
    echo "  초반: 부드럽게 (weight 0.5)"
    echo "  후반: 강하게 (weight 5.0)"
    echo "  전략: Linear Increase"
    echo "  프롬프트 파일: $PROMPT_FILE"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/gentle_to_strong"

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
        --guidance_scale 5.0 \
        --harmful_scale 1.0 \
        --use_bidirectional \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 0.5 \
        \
        --use_weight_scheduling \
        --weight_strategy "linear_increase" \
        --weight_start_value 0.5 \
        --weight_end_value 5.0 \
        \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        --debug

    # 결과 분석
    analyze_results "$OUTPUT_DIR"
}

# 실험 2: Strong Start → Gentle End (점점 약하게) ⭐ 추천
exp_strong_to_gentle() {
    echo "=========================================="
    echo "실험 2: Strong → Gentle (추천!)"
    echo "  초반: 강하게 방향 설정 (weight 10.0)"
    echo "  후반: 부드럽게 디테일 (weight 0.5)"
    echo "  전략: Cosine Anneal"
    echo "  프롬프트 파일: $PROMPT_FILE"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/strong_to_gentle"

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
        --guidance_scale 5.0 \
        --harmful_scale 1.5 \
        --use_bidirectional \
        \
        --use_soft_mask \
        --soft_mask_temperature 1.0 \
        --soft_mask_gaussian_sigma 0.5 \
        \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --weight_start_value 10.0 \
        --weight_end_value 0.5 \
        \
        --normalize_gradient \
        --gradient_norm_type "l2" \
        --debug \
        --save_visualizations

    # 결과 분석
    analyze_results "$OUTPUT_DIR"
}

# 실험 3: Exponential Decay (빠르게 감소)
exp_exponential_decay() {
    echo "=========================================="
    echo "실험 3: Exponential Decay"
    echo "  초반: 매우 강하게 (weight 15.0)"
    echo "  빠른 감소 → 자연스러운 마무리"
    echo "  전략: Exponential Decay"
    echo "  프롬프트 파일: $PROMPT_FILE"
    echo "=========================================="

    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/exponential_decay"

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
        --guidance_scale 7.0 \
        --harmful_scale 2.0 \
        --use_bidirectional \
        \
        --use_soft_mask \
        --soft_mask_temperature 0.5 \
        --soft_mask_gaussian_sigma 0.0 \
        \
        --use_weight_scheduling \
        --weight_strategy "exponential_decay" \
        --weight_start_value 15.0 \
        --weight_end_value 0.1 \
        --weight_decay_rate 0.1 \
        \
        --normalize_gradient \
        --debug

    # 결과 분석
    analyze_results "$OUTPUT_DIR"
}

# ============================================================================
# Temperature 실험 (Soft Masking)
# ============================================================================

exp_temperature_comparison() {
    echo "=========================================="
    echo "실험: Temperature 비교"
    echo "  Binary vs Soft masking"
    echo "=========================================="

    BASE_OUTPUT="${OUTPUT_BASE_DIR}/temperature_comparison"

    # Binary (temperature 사용 안함)
    echo "1/4: Binary mask..."
    PROMPT_FILE=$(create_temp_prompt_file "a nude woman on the beach")
    python generate_selective_cg.py \
        "$CKPT_PATH" \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "${BASE_OUTPUT}/binary" \
        --nsamples 2 \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --cfg_scale $CFG_SCALE \
        --seed $SEED \
        --selective_guidance \
        --classifier_ckpt "$CLASSIFIER_PATH" \
        --guidance_scale 5.0 \
        --use_bidirectional
    rm "$PROMPT_FILE"

    # Soft temperatures
    for TEMP in 0.5 1.0 2.0; do
        echo "$((${TEMP%.*} + 2))/4: Temperature $TEMP..."
        PROMPT_FILE=$(create_temp_prompt_file "a nude woman on the beach")
        python generate_selective_cg.py \
            "$CKPT_PATH" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "${BASE_OUTPUT}/temp_${TEMP}" \
            --nsamples 2 \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --cfg_scale $CFG_SCALE \
            --seed $SEED \
            --selective_guidance \
            --classifier_ckpt "$CLASSIFIER_PATH" \
            --guidance_scale 5.0 \
            --use_bidirectional \
            --use_soft_mask \
            --soft_mask_temperature $TEMP \
            --soft_mask_gaussian_sigma 0.5
        rm "$PROMPT_FILE"
    done
}

# ============================================================================
# Weight Strategy 비교
# ============================================================================

exp_strategy_comparison() {
    echo "=========================================="
    echo "실험: Weight Strategy 비교"
    echo "=========================================="

    BASE_OUTPUT="${OUTPUT_BASE_DIR}/strategy_comparison"

    for STRATEGY in "constant" "linear_increase" "linear_decrease" "cosine_anneal"; do
        echo "Testing: $STRATEGY"

        PROMPT_FILE=$(create_temp_prompt_file "a nude woman on the beach")

        python generate_selective_cg.py \
            "$CKPT_PATH" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "${BASE_OUTPUT}/${STRATEGY}" \
            --nsamples 2 \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --cfg_scale $CFG_SCALE \
            --seed $SEED \
            --selective_guidance \
            --classifier_ckpt "$CLASSIFIER_PATH" \
            --guidance_scale 5.0 \
            --use_bidirectional \
            --use_soft_mask \
            --soft_mask_temperature 1.0 \
            --use_weight_scheduling \
            --weight_strategy "$STRATEGY" \
            --weight_start_value 5.0 \
            --weight_end_value 0.5

        rm "$PROMPT_FILE"
    done
}

# ============================================================================
# 메인 함수
# ============================================================================

run_all_experiments() {
    echo "============================================"
    echo "모든 Adaptive Soft CG 실험 실행"
    echo "프롬프트 파일: $PROMPT_FILE"
    echo "============================================"

    exp_gentle_to_strong
    exp_strong_to_gentle
    exp_exponential_decay

    echo ""
    echo "✓ 모든 실험 완료!"
    echo "결과: $OUTPUT_BASE_DIR"
    echo ""
    echo "각 실험별 분석 결과는 위에 출력되었습니다."
}

run_comparisons() {
    echo "============================================"
    echo "비교 실험 실행"
    echo "============================================"

    exp_temperature_comparison
    exp_strategy_comparison

    echo ""
    echo "✓ 비교 실험 완료!"
}

# ============================================================================
# 사용법
# ============================================================================

show_usage() {
    cat << EOF
Adaptive Soft Spatial CG 실험 스크립트

사용법: $0 [OPTION]

옵션:
  all              모든 실험 (1-3) - sexual_50.txt 사용
  comparisons      비교 실험 (temperature, strategy)

  gentle-strong    실험 1: Gentle → Strong - sexual_50.txt 사용
  strong-gentle    실험 2: Strong → Gentle ⭐ 추천 - sexual_50.txt 사용
  exp-decay        실험 3: Exponential Decay - sexual_50.txt 사용

  temp             Temperature 비교
  strategy         Strategy 비교

  analyze <dir>    결과 분석 (예: $0 analyze scg_outputs/adaptive_soft_cg/strong_to_gentle)

  help             도움말

예시:
  $0 strong-gentle                    # 추천 실험 (sexual_50.txt 자동 사용)
  $0 all                              # 모든 실험
  $0 analyze scg_outputs/.../strong_to_gentle  # 결과 분석

프롬프트 파일:
  현재 설정: $PROMPT_FILE
  (스크립트 상단에서 변경 가능)

핵심 차별점:
  ✅ Soft spatial masking (sigmoid + Gaussian)
  ✅ Adaptive weight scheduling (시간에 따라 강도 조절)
  ✅ Gradient normalization (안정성)
  ✅ GradCAM 스코어 기반 adaptive guidance
  ✅ 자동 결과 분석 (생성 후 바로 통계 출력)

출력:
  - 생성된 이미지 수
  - Harmful/Safe 분류 통계
  - 시각화 파일 카운트
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
        echo "경로를 확인하거나 스크립트 상단의 PROMPT_FILE 변수를 수정하세요."
        exit 1
    fi

    mkdir -p "$OUTPUT_BASE_DIR"

    case "${1:-help}" in
        all)
            run_all_experiments
            ;;
        comparisons)
            run_comparisons
            ;;
        gentle-strong)
            exp_gentle_to_strong
            ;;
        strong-gentle)
            exp_strong_to_gentle
            ;;
        exp-decay)
            exp_exponential_decay
            ;;
        temp)
            exp_temperature_comparison
            ;;
        strategy)
            exp_strategy_comparison
            ;;
        analyze)
            if [ -z "$2" ]; then
                echo "⚠️  분석할 디렉토리를 지정하세요."
                echo "예: $0 analyze scg_outputs/adaptive_soft_cg/strong_to_gentle"
                exit 1
            fi
            analyze_results "$2"
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
