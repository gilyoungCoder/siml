#!/bin/bash
# ============================================================================
# Grid Search - Spatial CG (Nudity Safe Combined step_18900)
# GradCAM + Gaussian CDF threshold 방식
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
# 경로 설정 (Nudity Safe Combined step_18900)
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/nudity_three_class_safe_combined/checkpoint/step_18900/classifier.pth"
GRADCAM_STATS_FILE="./gradcam_nudity_stats_safe_combined.json"
OUTPUT_BASE_DIR="./scg_outputs/grid_search_spatial_nudity_safe_combined_step18900_norm"

# ============================================================================
# 고정 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=42
THRESHOLD_STRATEGY="cosine_anneal"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Grid Search 파라미터 정의
# ============================================================================

# GUIDANCE_SCALE: 유해 영역 가이던스 강도
GUIDANCE_SCALES=(10.0 15.0)

# SPATIAL_THRESHOLD: (start, end) 쌍
SPATIAL_THRESHOLDS=(
    "0.6,0.4"   # 높은 → 낮은
    "0.5,0.3"   # 높은 → 낮은
    "0.4,0.6"   # 낮은 → 높은
    "0.3,0.5"   # 낮은 → 높은
)

# HARMFUL_SCALE: 유해 반발력 배율
HARMFUL_SCALES=(1.0 1.5 2.0)

# BASE_GUIDANCE_SCALE: 비유해 영역 가이던스
BASE_GUIDANCE_SCALES=(1.0 2.0)

# ============================================================================
# 프롬프트 파일
# ============================================================================

PROMPT_FILE="./prompts/sexual_50.txt"

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

run_single_experiment() {
    local GS=$1
    local ST_START=$2
    local ST_END=$3
    local HS=$4
    local BGS=$5
    local EXP_NUM=$6
    local TOTAL_EXP=$7

    local EXPERIMENT_NAME="gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}"
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"

    print_header "[${EXP_NUM}/${TOTAL_EXP}] ${EXPERIMENT_NAME}"

    print_info "파라미터:"
    print_config "GUIDANCE_SCALE" "$GS"
    print_config "SPATIAL_THRESHOLD" "${ST_START} → ${ST_END}"
    print_config "HARMFUL_SCALE" "$HS"
    print_config "BASE_GUIDANCE_SCALE" "$BGS"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # generate_always_adaptive_spatial_cg.py 사용
    python generate_always_adaptive_spatial_cg_norm.py \
        $CKPT_PATH \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --nsamples $NSAMPLES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        --classifier_ckpt "$CLASSIFIER_PATH" \
        --gradcam_stats_file "$GRADCAM_STATS_FILE" \
        --guidance_scale $GS \
        --spatial_threshold_start $ST_START \
        --spatial_threshold_end $ST_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --guidance_start_step $GUIDANCE_START_STEP \
        --guidance_end_step $GUIDANCE_END_STEP \
        --harmful_scale $HS \
        --base_guidance_scale $BGS \
        --use_bidirectional

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_info "✓ 완료: ${EXPERIMENT_NAME}"
        echo "$EXPERIMENT_NAME,success" >> "${OUTPUT_BASE_DIR}/results.csv"
        return 0
    else
        print_error "✗ 실패: ${EXPERIMENT_NAME}"
        echo "$EXPERIMENT_NAME,failed" >> "${OUTPUT_BASE_DIR}/results.csv"
        return 1
    fi
}

# ============================================================================
# 메인 실행
# ============================================================================

main() {
    print_header "Grid Search - Spatial CG (Nudity Safe Combined step_18900)"

    # 결과 디렉토리 생성
    mkdir -p "$OUTPUT_BASE_DIR"

    # CSV 헤더 생성
    echo "experiment,status" > "${OUTPUT_BASE_DIR}/results.csv"

    # 총 실험 수 계산
    local TOTAL_EXPERIMENTS=$((${#GUIDANCE_SCALES[@]} * ${#SPATIAL_THRESHOLDS[@]} * ${#HARMFUL_SCALES[@]} * ${#BASE_GUIDANCE_SCALES[@]}))

    print_info "Grid Search 설정:"
    print_config "GUIDANCE_SCALES" "${GUIDANCE_SCALES[*]}"
    print_config "SPATIAL_THRESHOLDS" "${#SPATIAL_THRESHOLDS[@]} combinations"
    print_config "HARMFUL_SCALES" "${HARMFUL_SCALES[*]}"
    print_config "BASE_GUIDANCE_SCALES" "${BASE_GUIDANCE_SCALES[*]}"
    print_config "총 실험 수" "$TOTAL_EXPERIMENTS"
    echo ""

    print_info "Classifier 설정:"
    print_config "Classifier" "$CLASSIFIER_PATH"
    print_config "GradCAM Stats" "$GRADCAM_STATS_FILE"
    print_config "Prompts" "$PROMPT_FILE"
    echo ""

    print_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
    echo ""

    # 실험 카운터
    local EXP_NUM=0
    local SUCCESS_COUNT=0
    local FAIL_COUNT=0

    # Grid Search 루프
    for GS in "${GUIDANCE_SCALES[@]}"; do
        for ST in "${SPATIAL_THRESHOLDS[@]}"; do
            local ST_START=$(echo $ST | cut -d',' -f1)
            local ST_END=$(echo $ST | cut -d',' -f2)

            for HS in "${HARMFUL_SCALES[@]}"; do
                for BGS in "${BASE_GUIDANCE_SCALES[@]}"; do
                    EXP_NUM=$((EXP_NUM + 1))

                    if run_single_experiment $GS $ST_START $ST_END $HS $BGS $EXP_NUM $TOTAL_EXPERIMENTS; then
                        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                    else
                        FAIL_COUNT=$((FAIL_COUNT + 1))
                    fi

                    echo ""
                done
            done
        done
    done

    # 최종 결과 요약
    print_header "Grid Search 완료"

    echo -e "${GREEN}✓ 성공:${NC} $SUCCESS_COUNT / $TOTAL_EXPERIMENTS"
    [ $FAIL_COUNT -gt 0 ] && echo -e "${RED}✗ 실패:${NC} $FAIL_COUNT / $TOTAL_EXPERIMENTS"

    echo -e "\n${CYAN}결과 디렉토리:${NC} $OUTPUT_BASE_DIR"
    echo -e "${CYAN}결과 CSV:${NC} ${OUTPUT_BASE_DIR}/results.csv"

    local TOTAL_IMAGES=$(find "$OUTPUT_BASE_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo -e "${GREEN}총 ${TOTAL_IMAGES}개 이미지 생성${NC}"

    print_header "Grid Search 종료"
}

# ============================================================================
# 스크립트 시작
# ============================================================================

if [ ! -f "$PROMPT_FILE" ]; then
    print_error "프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$CLASSIFIER_PATH" ]; then
    print_error "분류기를 찾을 수 없습니다: $CLASSIFIER_PATH"
    exit 1
fi

if [ ! -f "$GRADCAM_STATS_FILE" ]; then
    print_error "GradCAM stats 파일을 찾을 수 없습니다: $GRADCAM_STATS_FILE"
    print_info "먼저 compute_gradcam_stats.py를 실행하여 stats 파일을 생성하세요"
    print_info "예: python compute_gradcam_stats.py --classifier_ckpt $CLASSIFIER_PATH --output_file $GRADCAM_STATS_FILE"
    exit 1
fi

main

exit 0
