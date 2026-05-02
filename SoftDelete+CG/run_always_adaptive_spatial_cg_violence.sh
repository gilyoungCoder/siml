#!/bin/bash
# ============================================================================
# Always-On Adaptive Spatial CG - Violence
# Violence concept 억제를 위한 이미지 생성
# Clustering centroid 기반 mode-aware guidance 적용
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

set -e

# ============================================================================
# 색상 정의
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ============================================================================
# 경로 설정
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
OUTPUT_BASE_DIR="./scg_outputs/violence_adaptive_spatial_cg"

# ============================================================================
# 기본 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1  # 프롬프트당 이미지 수
SEED=42

# ============================================================================
# Adaptive Spatial CG 파라미터
# ============================================================================

GUIDANCE_SCALE=10.0                   # 가이던스 강도
SPATIAL_THRESHOLD_START=0.45          # 초기: 상위 25% 영역만 (보수적 시작)
SPATIAL_THRESHOLD_END=0.75            # 최종: 평균 이상 영역 (균형)
THRESHOLD_STRATEGY="cosine_anneal"    # cosine_anneal (초반/후반 완만, 중간 급격)

# 선택적 파라미터
USE_BIDIRECTIONAL="--use_bidirectional"  # 양방향 가이던스 활성화
HARMFUL_SCALE=1.5                        # 유해 반발력
BASE_GUIDANCE_SCALE=2.0                  # 비유해 영역 약한 가이던스
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Visualization
SAVE_VISUALIZATIONS="--save_visualizations"  # 시각화 저장 활성화

# GradCAM Statistics (Violence용)
GRADCAM_STATS_FILE="./gradcam_violence_stats.json"

# ============================================================================
# 프롬프트 파일 설정
# ============================================================================

PROMPT_FILES=(
    "./prompts/violence_50.txt"
)

# ============================================================================
# 로그 함수
# ============================================================================

print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_config() {
    echo -e "${MAGENTA}  ├─${NC} $1: ${CYAN}$2${NC}"
}

# ============================================================================
# 실험 실행 함수
# ============================================================================

run_experiment() {
    local PROMPT_FILE="$1"
    local EXPERIMENT_NAME="$2"

    if [ ! -f "$PROMPT_FILE" ]; then
        print_error "프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
        return 1
    fi

    # 프롬프트 개수 계산
    local TOTAL_PROMPTS=$(grep -v '^\s*$' "$PROMPT_FILE" | grep -v '^\s*#' | wc -l)

    # 출력 디렉토리 생성
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    print_header "실험 시작: ${EXPERIMENT_NAME}"

    print_info "설정 정보:"
    print_config "프롬프트 파일" "$PROMPT_FILE"
    print_config "프롬프트 개수" "$TOTAL_PROMPTS"
    print_config "출력 디렉토리" "$OUTPUT_DIR"
    print_config "모델" "$CKPT_PATH"
    print_config "분류기" "$CLASSIFIER_PATH"
    echo ""

    print_info "생성 파라미터:"
    print_config "Inference Steps" "$NUM_INFERENCE_STEPS"
    print_config "CFG Scale" "$CFG_SCALE"
    print_config "Samples/Prompt" "$NSAMPLES"
    print_config "Seed" "$SEED"
    echo ""

    print_info "Adaptive Spatial CG 파라미터:"
    print_config "GradCAM Stats" "$GRADCAM_STATS_FILE"
    print_config "Guidance Scale (harmful)" "$GUIDANCE_SCALE"
    print_config "Base Guidance Scale (non-harmful)" "$BASE_GUIDANCE_SCALE"
    print_config "Spatial Threshold" "$SPATIAL_THRESHOLD_START → $SPATIAL_THRESHOLD_END"
    print_config "Threshold Strategy" "$THRESHOLD_STRATEGY"
    print_config "Bidirectional" "$([ -n "$USE_BIDIRECTIONAL" ] && echo 'Yes' || echo 'No')"
    [ -n "$USE_BIDIRECTIONAL" ] && print_config "Harmful Scale" "$HARMFUL_SCALE"
    print_config "Active Steps" "$GUIDANCE_START_STEP ~ $GUIDANCE_END_STEP"
    echo ""

    # 실행 명령어 구성
    local CMD="python generate_always_adaptive_spatial_cg.py \
        $CKPT_PATH \
        --prompt_file \"$PROMPT_FILE\" \
        --output_dir \"$OUTPUT_DIR\" \
        --nsamples $NSAMPLES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        --classifier_ckpt \"$CLASSIFIER_PATH\" \
        --gradcam_stats_file \"$GRADCAM_STATS_FILE\" \
        --guidance_scale $GUIDANCE_SCALE \
        --spatial_threshold_start $SPATIAL_THRESHOLD_START \
        --spatial_threshold_end $SPATIAL_THRESHOLD_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --guidance_start_step $GUIDANCE_START_STEP \
        --guidance_end_step $GUIDANCE_END_STEP \
        --harmful_scale $HARMFUL_SCALE \
        --base_guidance_scale $BASE_GUIDANCE_SCALE \
        $USE_BIDIRECTIONAL \
        $SAVE_VISUALIZATIONS"

    print_info "실행 명령어:"
    echo -e "${CYAN}$CMD${NC}\n"

    # 실행
    eval $CMD

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_info "✓ 실험 완료: ${EXPERIMENT_NAME}"
        print_info "✓ 결과 저장: ${OUTPUT_DIR}"
    else
        print_error "✗ 실험 실패: ${EXPERIMENT_NAME} (Exit code: $EXIT_CODE)"
        return $EXIT_CODE
    fi

    echo ""
}

# ============================================================================
# 메인 실행
# ============================================================================

main() {
    print_header "Always-On Adaptive Spatial CG - Violence 실험 시작"

    print_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
    echo ""

    # 실험 카운터
    local TOTAL_EXPERIMENTS=${#PROMPT_FILES[@]}
    local SUCCESS_COUNT=0
    local FAIL_COUNT=0

    print_info "총 ${TOTAL_EXPERIMENTS}개 실험 예정\n"

    # 각 프롬프트 파일에 대해 실험 실행
    for i in "${!PROMPT_FILES[@]}"; do
        local PROMPT_FILE="${PROMPT_FILES[$i]}"
        local BASENAME=$(basename "$PROMPT_FILE" .txt)
        local EXPERIMENT_NAME="${BASENAME}_gs${GUIDANCE_SCALE}_st${SPATIAL_THRESHOLD_START}-${SPATIAL_THRESHOLD_END}_${THRESHOLD_STRATEGY}"

        print_info "[$(($i + 1))/$TOTAL_EXPERIMENTS] 실험: $EXPERIMENT_NAME"

        if run_experiment "$PROMPT_FILE" "$EXPERIMENT_NAME"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    # 최종 결과 요약
    print_header "모든 실험 완료"

    echo -e "${GREEN}✓ 성공:${NC} $SUCCESS_COUNT / $TOTAL_EXPERIMENTS"
    [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}✗ 실패:${NC} $FAIL_COUNT / $TOTAL_EXPERIMENTS"

    echo -e "\n${CYAN}결과 디렉토리:${NC} $OUTPUT_BASE_DIR"
    echo ""

    # 결과 미리보기
    if command -v tree &> /dev/null; then
        print_info "결과 구조:"
        tree -L 2 "$OUTPUT_BASE_DIR" 2>/dev/null || ls -R "$OUTPUT_BASE_DIR"
    else
        print_info "생성된 파일:"
        find "$OUTPUT_BASE_DIR" -name "*.png" | head -10
        local TOTAL_IMAGES=$(find "$OUTPUT_BASE_DIR" -name "*.png" | wc -l)
        echo -e "${GREEN}  총 ${TOTAL_IMAGES}개 이미지 생성${NC}"
    fi

    print_header "실험 종료"
}

# ============================================================================
# 스크립트 시작
# ============================================================================

# 디렉토리 확인
if [ ! -d "prompts" ]; then
    print_error "prompts 디렉토리를 찾을 수 없습니다."
    exit 1
fi

if [ ! -f "$CLASSIFIER_PATH" ]; then
    print_error "분류기를 찾을 수 없습니다: $CLASSIFIER_PATH"
    exit 1
fi

if [ ! -f "$GRADCAM_STATS_FILE" ]; then
    print_warn "GradCAM stats 파일을 찾을 수 없습니다: $GRADCAM_STATS_FILE"
    print_warn "GradCAM normalization 없이 실행됩니다."
fi

# 메인 실행
main

exit 0
