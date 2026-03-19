#!/bin/bash
# ============================================================================
# Multi-Concept Always-On Adaptive Spatial CG - Run Script
# ============================================================================

export CUDA_VISIBLE_DEVICES=7

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
OUTPUT_BASE_DIR="./scg_outputs/multi_concept_always_adaptive_cg"

# Classifier paths
NUDITY_CLASSIFIER="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
VIOLENCE_CLASSIFIER="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
VANGOGH_CLASSIFIER="./work_dirs/vangogh_three_class_diff/checkpoint/step_7200/classifier.pth"

# ============================================================================
# 기본 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1  # 프롬프트당 이미지 수
SEED=42

# ============================================================================
# Adaptive Spatial CG 파라미터 (간소화된 버전!)
# ============================================================================

GUIDANCE_SCALE=10.0                  # 가이던스 강도
SPATIAL_THRESHOLD_START=0.4         # 초기 스텝 임계값
SPATIAL_THRESHOLD_END=0.1           # 최종 스텝 임계값
THRESHOLD_STRATEGY="cosine_anneal"  # linear_decrease, cosine_anneal, constant

# 선택적 파라미터
USE_BIDIRECTIONAL="--use_bidirectional"  # 양방향 가이던스 활성화
HARMFUL_SCALE=1.5                        # 유해 반발력 강도
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Visualization
SAVE_VISUALIZATIONS=""  # "--save_visualizations" to enable

# ============================================================================
# 실험 구성 (개념 조합 선택)
# ============================================================================

# 실험 1: Nudity + Violence
EXPERIMENT_1_NAME="nudity_violence"
EXPERIMENT_1_CONCEPTS="--nudity_enabled --violence_enabled"
EXPERIMENT_1_PROMPT_FILE="./prompts/sexual_50.txt"  # Mixed prompts

# 실험 2: Nudity + VanGogh
EXPERIMENT_2_NAME="nudity_vangogh"
EXPERIMENT_2_CONCEPTS="--nudity_enabled --vangogh_enabled"
EXPERIMENT_2_PROMPT_FILE="./prompts/sexual_50.txt"

# 실험 3: Violence + VanGogh
EXPERIMENT_3_NAME="vangogh_nudity"
EXPERIMENT_3_CONCEPTS="--nudity_enabled --vangogh_enabled"
EXPERIMENT_3_PROMPT_FILE="./prompts/violence_50.txt"

# # 실험 4: All Three Concepts
# EXPERIMENT_4_NAME="nudity_violence_vangogh"
# EXPERIMENT_4_CONCEPTS="--nudity_enabled --violence_enabled --vangogh_enabled"
# EXPERIMENT_4_PROMPT_FILE="./prompts/sexual_50.txt"

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
    local EXPERIMENT_NAME="$1"
    local CONCEPT_FLAGS="$2"
    local PROMPT_FILE="$3"

    if [ ! -f "$PROMPT_FILE" ]; then
        print_error "프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
        return 1
    fi

    # 출력 디렉토리 생성
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    print_header "실험: ${EXPERIMENT_NAME}"

    print_info "파라미터:"
    print_config "Concepts" "${CONCEPT_FLAGS}"
    print_config "Guidance Scale" "$GUIDANCE_SCALE"
    print_config "Spatial Threshold" "$SPATIAL_THRESHOLD_START → $SPATIAL_THRESHOLD_END"
    print_config "Strategy" "$THRESHOLD_STRATEGY"
    print_config "Bidirectional" "$([ -n "$USE_BIDIRECTIONAL" ] && echo 'Yes' || echo 'No')"
    print_config "Harmful Scale" "$HARMFUL_SCALE"
    print_config "Prompt File" "$PROMPT_FILE"
    echo ""

    # 실행 명령어 구성
    local CMD="python generate_multi_concept_always_adaptive_cg.py \
        $CKPT_PATH \
        --prompt_file \"$PROMPT_FILE\" \
        --output_dir \"$OUTPUT_DIR\" \
        --nsamples $NSAMPLES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        $CONCEPT_FLAGS \
        --nudity_classifier \"$NUDITY_CLASSIFIER\" \
        --violence_classifier \"$VIOLENCE_CLASSIFIER\" \
        --vangogh_classifier \"$VANGOGH_CLASSIFIER\" \
        --guidance_scale $GUIDANCE_SCALE \
        --spatial_threshold_start $SPATIAL_THRESHOLD_START \
        --spatial_threshold_end $SPATIAL_THRESHOLD_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --guidance_start_step $GUIDANCE_START_STEP \
        --guidance_end_step $GUIDANCE_END_STEP \
        --harmful_scale $HARMFUL_SCALE \
        $USE_BIDIRECTIONAL \
        $SAVE_VISUALIZATIONS"

    # 실행 시간 측정
    local START_TIME=$(date +%s)

    eval $CMD

    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        print_info "✓ 완료 (${DURATION}초)"
        echo "$EXPERIMENT_NAME,$CONCEPT_FLAGS,$DURATION,SUCCESS" >> "$OUTPUT_BASE_DIR/experiment_log.csv"
        return 0
    else
        print_error "✗ 실패 (Exit code: $EXIT_CODE)"
        echo "$EXPERIMENT_NAME,$CONCEPT_FLAGS,$DURATION,FAILED" >> "$OUTPUT_BASE_DIR/experiment_log.csv"
        return $EXIT_CODE
    fi
}

# ============================================================================
# 메인 함수
# ============================================================================

main() {
    print_header "Multi-Concept Always-On Adaptive Spatial CG"

    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_BASE_DIR"

    # CSV 로그 파일 초기화
    echo "experiment_name,concepts,duration_sec,status" > "$OUTPUT_BASE_DIR/experiment_log.csv"

    print_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
    echo ""

    # 실험 카운터
    local SUCCESS_COUNT=0
    local FAIL_COUNT=0
    local OVERALL_START_TIME=$(date +%s)

    # 실험 1: Nudity + Violence
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "실험 1/4: Nudity + Violence"
    if run_experiment "$EXPERIMENT_1_NAME" "$EXPERIMENT_1_CONCEPTS" "$EXPERIMENT_1_PROMPT_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    # 실험 2: Nudity + VanGogh
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "실험 2/4: Nudity + VanGogh"
    if run_experiment "$EXPERIMENT_2_NAME" "$EXPERIMENT_2_CONCEPTS" "$EXPERIMENT_2_PROMPT_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    # 실험 3: Violence + VanGogh
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "실험 3/4: Violence + VanGogh"
    if run_experiment "$EXPERIMENT_3_NAME" "$EXPERIMENT_3_CONCEPTS" "$EXPERIMENT_3_PROMPT_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    # 실험 4: All Three
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "실험 4/4: Nudity + Violence + VanGogh"
    if run_experiment "$EXPERIMENT_4_NAME" "$EXPERIMENT_4_CONCEPTS" "$EXPERIMENT_4_PROMPT_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    local OVERALL_END_TIME=$(date +%s)
    local TOTAL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
    local HOURS=$((TOTAL_DURATION / 3600))
    local MINUTES=$(((TOTAL_DURATION % 3600) / 60))
    local SECONDS=$((TOTAL_DURATION % 60))

    # 최종 결과 요약
    print_header "실험 완료"

    echo -e "${GREEN}✓ 성공:${NC} $SUCCESS_COUNT / 4"
    [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}✗ 실패:${NC} $FAIL_COUNT / 4"
    echo -e "${CYAN}⏱ 총 소요 시간:${NC} ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"

    echo -e "\n${CYAN}결과 디렉토리:${NC} $OUTPUT_BASE_DIR"
    echo -e "${CYAN}실험 로그:${NC} $OUTPUT_BASE_DIR/experiment_log.csv"
    echo ""

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

if [ ! -f "$NUDITY_CLASSIFIER" ]; then
    print_error "Nudity 분류기를 찾을 수 없습니다: $NUDITY_CLASSIFIER"
    exit 1
fi

if [ ! -f "$VIOLENCE_CLASSIFIER" ]; then
    print_error "Violence 분류기를 찾을 수 없습니다: $VIOLENCE_CLASSIFIER"
    exit 1
fi

if [ ! -f "$VANGOGH_CLASSIFIER" ]; then
    print_error "VanGogh 분류기를 찾을 수 없습니다: $VANGOGH_CLASSIFIER"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_BASE_DIR"

# 시작 시간 기록
echo "Multi-Concept Experiments 시작: $(date)" > "$OUTPUT_BASE_DIR/start_time.txt"

# 메인 실행
main

# 종료 시간 기록
echo "Multi-Concept Experiments 종료: $(date)" >> "$OUTPUT_BASE_DIR/start_time.txt"

exit 0
