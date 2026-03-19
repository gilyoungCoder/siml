#!/bin/bash
# ============================================================================
# Cluster-Aware Adaptive Spatial CG - Nudity Grayscale Version
# Grayscale classifier로 학습된 모델 사용
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
# 경로 설정 (Nudity Grayscale)
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/nudity_three_class_grayscale/checkpoint/step_11200/classifier.pth"
CENTROIDS_PATH="./cluster_centroids/nudity_grayscale_clusters.pt"
GRADCAM_STATS_FILE="./gradcam_nudity_stats_grayscale.json"
OUTPUT_BASE_DIR="./scg_outputs/cluster_spatial_cg_nudity_grayscale"

# ============================================================================
# 기본 생성 파라미터
# ============================================================================

NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=42

# ============================================================================
# Cluster-Aware Spatial CG 파라미터
# ============================================================================

# Guidance 강도
GUIDANCE_SCALE=10.0
HARMFUL_SCALE=1.5
BASE_GUIDANCE_SCALE=2.0

# Spatial threshold (adaptive)
SPATIAL_THRESHOLD_START=0.4
SPATIAL_THRESHOLD_END=0.7
THRESHOLD_STRATEGY="cosine_anneal"

# Active step range
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Bidirectional guidance
USE_BIDIRECTIONAL="--use_bidirectional"

# Visualization
SAVE_VISUALIZATIONS="--save_visualizations"

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_config() {
    echo -e "${MAGENTA}  ├─${NC} $1: ${CYAN}$2${NC}"
}

# ============================================================================
# 메인 실행
# ============================================================================

main() {
    print_header "Cluster-Aware Spatial CG (Nudity Grayscale)"

    # GPU 정보
    print_info "GPU 정보:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %s (%s total, %s free)\n", $1, $2, $3, $4}'
    echo ""

    # 설정 출력
    print_info "설정 정보:"
    print_config "Model" "$CKPT_PATH"
    print_config "Classifier (Grayscale)" "$CLASSIFIER_PATH"
    print_config "Centroids" "$CENTROIDS_PATH"
    print_config "GradCAM Stats" "$GRADCAM_STATS_FILE"
    print_config "Prompt File" "$PROMPT_FILE"
    print_config "Output Dir" "$OUTPUT_BASE_DIR"
    echo ""

    print_info "Guidance 파라미터:"
    print_config "Guidance Scale" "$GUIDANCE_SCALE"
    print_config "Harmful Scale" "$HARMFUL_SCALE"
    print_config "Base Guidance Scale" "$BASE_GUIDANCE_SCALE"
    print_config "Spatial Threshold" "$SPATIAL_THRESHOLD_START -> $SPATIAL_THRESHOLD_END"
    print_config "Strategy" "$THRESHOLD_STRATEGY"
    print_config "Bidirectional" "$([ -n "$USE_BIDIRECTIONAL" ] && echo 'Yes' || echo 'No')"
    echo ""

    # 출력 디렉토리
    EXPERIMENT_NAME="gs${GUIDANCE_SCALE}_st${SPATIAL_THRESHOLD_START}-${SPATIAL_THRESHOLD_END}_hs${HARMFUL_SCALE}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    print_info "실험 이름: $EXPERIMENT_NAME"
    print_info "출력 디렉토리: $OUTPUT_DIR"
    echo ""

    # 실행 명령어
    CMD="python generate_cluster_spatial_cg.py \
        $CKPT_PATH \
        --prompt_file \"$PROMPT_FILE\" \
        --output_dir \"$OUTPUT_DIR\" \
        --nsamples $NSAMPLES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        --classifier_ckpt \"$CLASSIFIER_PATH\" \
        --centroids_path \"$CENTROIDS_PATH\" \
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

    if [ $? -eq 0 ]; then
        print_header "실험 완료"
        print_info "결과: $OUTPUT_DIR"

        TOTAL_IMAGES=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" | wc -l)
        print_info "생성된 이미지: $TOTAL_IMAGES"
    else
        print_error "실험 실패"
        exit 1
    fi
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

# GradCAM stats 체크
if [ ! -f "$GRADCAM_STATS_FILE" ]; then
    print_error "GradCAM stats 파일을 찾을 수 없습니다: $GRADCAM_STATS_FILE"
    print_info "먼저 compute_gradcam_stats_nudity_grayscale.sh를 실행하세요:"
    echo ""
    echo "  ./compute_gradcam_stats_nudity_grayscale.sh"
    echo ""
    exit 1
fi

# Centroids 체크 (없으면 경고만)
if [ ! -f "$CENTROIDS_PATH" ]; then
    print_error "클러스터 센트로이드를 찾을 수 없습니다: $CENTROIDS_PATH"
    print_info "클러스터링을 먼저 실행하거나, 기존 centroids 경로를 수정하세요."
    exit 1
fi

main

exit 0
