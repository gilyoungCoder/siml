#!/bin/bash
# ============================================================================
# Grid Search for Spatial Threshold - VanGogh (Gaussian CDF Normalized)
# SPATIAL_THRESHOLD_START / SPATIAL_THRESHOLD_END 조합 탐색
# ============================================================================

# Conda 환경 활성화 (nohup 실행 시 필요)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd

export CUDA_VISIBLE_DEVICES=6

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
# 경로 설정 - VanGogh
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/vangogh_three_class_diff/checkpoint/step_7200/classifier.pth"
OUTPUT_BASE_DIR="./scg_outputs/grid_search_threshold_vangogh"
GRADCAM_STATS_FILE="./gradcam_vangogh_stats.json"

# ============================================================================
# 기본 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=42

# ============================================================================
# 고정 파라미터
# ============================================================================

GUIDANCE_SCALE=10.0
THRESHOLD_STRATEGY="cosine_anneal"
USE_BIDIRECTIONAL="--use_bidirectional"
HARMFUL_SCALE=1.5
BASE_GUIDANCE_SCALE=2.0
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Grid Search 파라미터 (Gaussian CDF 기준 0~1 범위)
# ============================================================================

THRESHOLD_STARTS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
THRESHOLD_ENDS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# ============================================================================
# 프롬프트 파일 - VanGogh
# ============================================================================

PROMPT_FILE="./prompts/vangogh_50.txt"

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

# ============================================================================
# Stats 파일 체크
# ============================================================================

if [ ! -f "$GRADCAM_STATS_FILE" ]; then
    print_error "GradCAM stats 파일이 없습니다: $GRADCAM_STATS_FILE"
    print_info "먼저 다음 명령을 실행하세요:"
    echo -e "  ${CYAN}./compute_gradcam_stats_vangogh.sh${NC}"
    exit 1
fi

# ============================================================================
# 결과 저장용 CSV 초기화
# ============================================================================

RESULTS_CSV="${OUTPUT_BASE_DIR}/grid_search_results.csv"
mkdir -p "$OUTPUT_BASE_DIR"
echo "threshold_start,threshold_end,output_dir,status" > "$RESULTS_CSV"

# ============================================================================
# Grid Search 실행
# ============================================================================

print_header "Grid Search: VanGogh Spatial Threshold (Gaussian CDF Normalized)"

print_info "탐색 범위:"
echo -e "  ${CYAN}THRESHOLD_START:${NC} ${THRESHOLD_STARTS[*]}"
echo -e "  ${CYAN}THRESHOLD_END:${NC} ${THRESHOLD_ENDS[*]}"

TOTAL_COMBINATIONS=$((${#THRESHOLD_STARTS[@]} * ${#THRESHOLD_ENDS[@]}))
print_info "총 ${TOTAL_COMBINATIONS}개 조합 탐색 예정"
echo ""

CURRENT=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for START in "${THRESHOLD_STARTS[@]}"; do
    for END in "${THRESHOLD_ENDS[@]}"; do
        CURRENT=$((CURRENT + 1))

        # START > END인 경우 스킵 (의미 없는 조합)
        if (( $(echo "$START > $END" | bc -l) )); then
            print_warn "[$CURRENT/$TOTAL_COMBINATIONS] 스킵: START($START) > END($END)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo "$START,$END,SKIPPED,invalid_combination" >> "$RESULTS_CSV"
            continue
        fi

        EXPERIMENT_NAME="start${START}_end${END}"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"

        print_header "[$CURRENT/$TOTAL_COMBINATIONS] START=$START, END=$END"

        mkdir -p "$OUTPUT_DIR"

        # 실행 명령어
        CMD="python generate_always_adaptive_spatial_cg.py \
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
            --spatial_threshold_start $START \
            --spatial_threshold_end $END \
            --threshold_strategy $THRESHOLD_STRATEGY \
            --guidance_start_step $GUIDANCE_START_STEP \
            --guidance_end_step $GUIDANCE_END_STEP \
            --harmful_scale $HARMFUL_SCALE \
            --base_guidance_scale $BASE_GUIDANCE_SCALE \
            $USE_BIDIRECTIONAL"

        print_info "실행: $EXPERIMENT_NAME"

        if eval $CMD; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "$START,$END,$OUTPUT_DIR,success" >> "$RESULTS_CSV"
            print_info "완료: $EXPERIMENT_NAME"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "$START,$END,$OUTPUT_DIR,failed" >> "$RESULTS_CSV"
            print_error "실패: $EXPERIMENT_NAME"
        fi

        echo ""
    done
done

# ============================================================================
# 최종 결과 요약
# ============================================================================

print_header "Grid Search 완료 - VanGogh"

echo -e "${GREEN}성공:${NC} $SUCCESS_COUNT"
echo -e "${YELLOW}스킵:${NC} $SKIP_COUNT (START > END)"
echo -e "${RED}실패:${NC} $FAIL_COUNT"
echo ""
echo -e "${CYAN}결과 CSV:${NC} $RESULTS_CSV"
echo -e "${CYAN}출력 디렉토리:${NC} $OUTPUT_BASE_DIR"

print_header "Grid Search 종료"
