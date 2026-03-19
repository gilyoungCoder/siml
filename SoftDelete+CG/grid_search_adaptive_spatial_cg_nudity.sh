#!/bin/bash
# ============================================================================
# Grid Search for Adaptive Spatial CG - Nudity
# 여러 hyperparameter 조합을 자동으로 실험 (Nudity concept)
# Usage: nohup bash grid_search_adaptive_spatial_cg_nudity.sh > grid_search_nudity.log 2>&1 &
# ============================================================================

export CUDA_VISIBLE_DEVICES=4

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
CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
OUTPUT_BASE_DIR="./scg_outputs/grid_search_adaptive_spatial_cg_nudity"

# ============================================================================
# 기본 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1  # 프롬프트당 이미지 수
SEED=42

# ============================================================================
# Grid Search 파라미터 범위
# ============================================================================

# Guidance Scale 후보
GUIDANCE_SCALES=(5.0 7.0 10.0 15.0 20.0)

# Harmful Scale 후보
HARMFUL_SCALES=(1.0 1.5 2.0)

# Spatial Threshold 후보 (start-end 쌍)
SPATIAL_THRESHOLD_PAIRS=(
    "0.5:0.1"  # 높게 시작 → 낮게
    "0.4:0.1"
    "0.3:0.1"
    "0.4:0.2"  # 중간 범위
    "0.1:0.5"  # 낮게 시작 → 높게 (역순)
    "0.1:0.4"
    "0.2:0.5"
)

# Threshold Strategy 후보
THRESHOLD_STRATEGIES=("linear_decrease" "cosine_anneal")

# Guidance Step 범위 후보 (start:end)
GUIDANCE_STEP_RANGES=(
    "0:50"   # 전체 구간
)

# ============================================================================
# 프롬프트 파일 설정
# ============================================================================

PROMPT_FILES=(
    "./prompts/sexual_50.txt"
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
    local GUIDANCE_SCALE="$2"
    local HARMFUL_SCALE="$3"
    local SPATIAL_THRESHOLD_START="$4"
    local SPATIAL_THRESHOLD_END="$5"
    local THRESHOLD_STRATEGY="$6"
    local GUIDANCE_START_STEP="$7"
    local GUIDANCE_END_STEP="$8"

    if [ ! -f "$PROMPT_FILE" ]; then
        print_error "프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
        return 1
    fi

    # 프롬프트 개수 계산
    local TOTAL_PROMPTS=$(grep -v '^\s*$' "$PROMPT_FILE" | grep -v '^\s*#' | wc -l)
    local BASENAME=$(basename "$PROMPT_FILE" .txt)

    # 실험 이름 생성
    local EXPERIMENT_NAME="${BASENAME}_gs${GUIDANCE_SCALE}_hs${HARMFUL_SCALE}_st${SPATIAL_THRESHOLD_START}-${SPATIAL_THRESHOLD_END}_${THRESHOLD_STRATEGY}_steps${GUIDANCE_START_STEP}-${GUIDANCE_END_STEP}"

    # 출력 디렉토리 생성
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    print_header "실험: ${EXPERIMENT_NAME}"

    print_info "파라미터:"
    print_config "Guidance Scale" "$GUIDANCE_SCALE"
    print_config "Harmful Scale" "$HARMFUL_SCALE"
    print_config "Spatial Threshold" "$SPATIAL_THRESHOLD_START → $SPATIAL_THRESHOLD_END"
    print_config "Strategy" "$THRESHOLD_STRATEGY"
    print_config "Active Steps" "$GUIDANCE_START_STEP ~ $GUIDANCE_END_STEP"
    print_config "Prompts" "$TOTAL_PROMPTS"
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
        --guidance_scale $GUIDANCE_SCALE \
        --spatial_threshold_start $SPATIAL_THRESHOLD_START \
        --spatial_threshold_end $SPATIAL_THRESHOLD_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --guidance_start_step $GUIDANCE_START_STEP \
        --guidance_end_step $GUIDANCE_END_STEP \
        --harmful_scale $HARMFUL_SCALE \
        --use_bidirectional \
        --save_visualizations"

    # 실행 시간 측정
    local START_TIME=$(date +%s)

    eval $CMD

    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        print_info "✓ 완료 (${DURATION}초)"
        echo "$EXPERIMENT_NAME,$GUIDANCE_SCALE,$HARMFUL_SCALE,$SPATIAL_THRESHOLD_START,$SPATIAL_THRESHOLD_END,$THRESHOLD_STRATEGY,$GUIDANCE_START_STEP,$GUIDANCE_END_STEP,$DURATION,SUCCESS" >> "$OUTPUT_BASE_DIR/experiment_log.csv"
        return 0
    else
        print_error "✗ 실패 (Exit code: $EXIT_CODE)"
        echo "$EXPERIMENT_NAME,$GUIDANCE_SCALE,$HARMFUL_SCALE,$SPATIAL_THRESHOLD_START,$SPATIAL_THRESHOLD_END,$THRESHOLD_STRATEGY,$GUIDANCE_START_STEP,$GUIDANCE_END_STEP,$DURATION,FAILED" >> "$OUTPUT_BASE_DIR/experiment_log.csv"
        return $EXIT_CODE
    fi
}

# ============================================================================
# 메인 Grid Search 함수
# ============================================================================

main() {
    print_header "Grid Search for Adaptive Spatial CG - Nudity"

    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_BASE_DIR"

    # CSV 로그 파일 초기화
    echo "experiment_name,guidance_scale,harmful_scale,spatial_threshold_start,spatial_threshold_end,threshold_strategy,guidance_start_step,guidance_end_step,duration_sec,status" > "$OUTPUT_BASE_DIR/experiment_log.csv"

    print_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
    echo ""

    # Grid Search 조합 계산
    local TOTAL_COMBINATIONS=$((
        ${#PROMPT_FILES[@]} *
        ${#GUIDANCE_SCALES[@]} *
        ${#HARMFUL_SCALES[@]} *
        ${#SPATIAL_THRESHOLD_PAIRS[@]} *
        ${#THRESHOLD_STRATEGIES[@]} *
        ${#GUIDANCE_STEP_RANGES[@]}
    ))

    print_info "총 ${TOTAL_COMBINATIONS}개 조합 실험 예정"
    print_info "각 조합당 약 1-2분 소요 예상 (총 ${TOTAL_COMBINATIONS}분 ~ $((TOTAL_COMBINATIONS * 2))분)"
    echo ""

    # 실험 카운터
    local EXPERIMENT_NUM=0
    local SUCCESS_COUNT=0
    local FAIL_COUNT=0
    local OVERALL_START_TIME=$(date +%s)

    # Grid Search 실행
    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
        for GS in "${GUIDANCE_SCALES[@]}"; do
            for HS in "${HARMFUL_SCALES[@]}"; do
                for ST_PAIR in "${SPATIAL_THRESHOLD_PAIRS[@]}"; do
                    ST_START=$(echo $ST_PAIR | cut -d':' -f1)
                    ST_END=$(echo $ST_PAIR | cut -d':' -f2)

                    for STRATEGY in "${THRESHOLD_STRATEGIES[@]}"; do
                        for STEP_RANGE in "${GUIDANCE_STEP_RANGES[@]}"; do
                            STEP_START=$(echo $STEP_RANGE | cut -d':' -f1)
                            STEP_END=$(echo $STEP_RANGE | cut -d':' -f2)

                            EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))

                            print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                            print_info "실험 [$EXPERIMENT_NUM/$TOTAL_COMBINATIONS]"

                            if run_experiment "$PROMPT_FILE" "$GS" "$HS" "$ST_START" "$ST_END" "$STRATEGY" "$STEP_START" "$STEP_END"; then
                                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                            else
                                FAIL_COUNT=$((FAIL_COUNT + 1))
                            fi

                            # 진행률 출력
                            local PROGRESS=$((EXPERIMENT_NUM * 100 / TOTAL_COMBINATIONS))
                            print_info "진행률: ${PROGRESS}% (성공: $SUCCESS_COUNT, 실패: $FAIL_COUNT)"
                            echo ""
                        done
                    done
                done
            done
        done
    done

    local OVERALL_END_TIME=$(date +%s)
    local TOTAL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
    local HOURS=$((TOTAL_DURATION / 3600))
    local MINUTES=$(((TOTAL_DURATION % 3600) / 60))
    local SECONDS=$((TOTAL_DURATION % 60))

    # 최종 결과 요약
    print_header "Grid Search 완료"

    echo -e "${GREEN}✓ 성공:${NC} $SUCCESS_COUNT / $TOTAL_COMBINATIONS"
    [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}✗ 실패:${NC} $FAIL_COUNT / $TOTAL_COMBINATIONS"
    echo -e "${CYAN}⏱ 총 소요 시간:${NC} ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"

    echo -e "\n${CYAN}결과 디렉토리:${NC} $OUTPUT_BASE_DIR"
    echo -e "${CYAN}실험 로그:${NC} $OUTPUT_BASE_DIR/experiment_log.csv"
    echo ""

    # 결과 통계
    print_info "생성된 이미지:"
    local TOTAL_IMAGES=$(find "$OUTPUT_BASE_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo -e "${GREEN}  총 ${TOTAL_IMAGES}개 이미지 생성${NC}"

    # 최적 파라미터 분석 (간단한 통계)
    print_info "실험 로그 저장: $OUTPUT_BASE_DIR/experiment_log.csv"
    print_info "결과 분석 명령어:"
    echo -e "${CYAN}  cat $OUTPUT_BASE_DIR/experiment_log.csv | grep SUCCESS | wc -l${NC}"

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

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_BASE_DIR"

# 시작 시간 기록
echo "Grid Search 시작: $(date)" > "$OUTPUT_BASE_DIR/start_time.txt"

# 메인 실행
main

# 종료 시간 기록
echo "Grid Search 종료: $(date)" >> "$OUTPUT_BASE_DIR/start_time.txt"

exit 0
