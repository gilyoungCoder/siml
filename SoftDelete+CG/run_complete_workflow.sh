#!/bin/bash
# ============================================================================
# Complete Workflow: Soft Spatial CG
# 전체 실험 워크플로우를 한번에 실행
# ============================================================================

set -e

# ============================================================================
# 색상 정의
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# 로깅 함수
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..60})${NC}"
}

# ============================================================================
# 환경 확인
# ============================================================================

check_environment() {
    log_section "환경 확인 중..."

    # Python 확인
    if ! command -v python &> /dev/null; then
        log_error "Python을 찾을 수 없습니다."
        exit 1
    fi
    log_info "Python: $(python --version)"

    # 필수 파일 확인
    REQUIRED_FILES=(
        "geo_utils/selective_guidance_utils.py"
        "configs/soft_spatial_cg_tuning.yaml"
        "configs/multi_concept_test_prompts.json"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "필수 파일을 찾을 수 없습니다: $file"
            exit 1
        fi
    done

    log_success "모든 필수 파일 존재"

    # GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU 사용 가능"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
    else
        log_warning "GPU를 감지할 수 없습니다. CPU로 실행됩니다."
    fi
}

# ============================================================================
# Classifier 확인
# ============================================================================

check_classifier() {
    log_section "Classifier 확인 중..."

    CLASSIFIER_PATH="${1:-checkpoints/nude_classifier_best.pth}"

    if [ ! -f "$CLASSIFIER_PATH" ]; then
        log_error "Classifier를 찾을 수 없습니다: $CLASSIFIER_PATH"
        log_info "Classifier를 학습하거나 경로를 지정하세요:"
        log_info "  export CLASSIFIER_PATH=/path/to/classifier.pth"
        exit 1
    fi

    log_success "Classifier 발견: $CLASSIFIER_PATH"
    export CLASSIFIER_PATH
}

# ============================================================================
# 워크플로우 단계들
# ============================================================================

# Step 1: 시각화 생성
step_visualize() {
    log_section "Step 1: Weight Scheduling 시각화"

    bash visualize_weight_schedules.py \
        --num_steps 50 \
        --output_dir visualizations

    log_success "시각화 완료"
}

# Step 2: 빠른 테스트
step_quick_test() {
    log_section "Step 2: 빠른 기능 테스트"

    log_info "기본 설정으로 단일 이미지 생성 테스트..."

    bash quick_test.sh default

    log_success "빠른 테스트 완료"
}

# Step 3: Preset 실험
step_presets() {
    log_section "Step 3: Preset 실험 실행"

    log_info "4가지 preset 설정으로 실험 중..."

    bash run_soft_cg_experiment.sh all-presets

    log_success "Preset 실험 완료"
}

# Step 4: 비교 실험
step_comparisons() {
    log_section "Step 4: 파라미터 비교 실험"

    log_info "Temperature, Strategy, Scale 비교 실험 중..."

    bash run_soft_cg_experiment.sh all-comparisons

    log_success "비교 실험 완료"
}

# Step 5: 결과 분석
step_analyze() {
    log_section "Step 5: 결과 분석 및 리포트 생성"

    bash analyze_results.sh all

    log_success "분석 완료"
}

# ============================================================================
# 메뉴 시스템
# ============================================================================

show_menu() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          Soft Spatial CG - Complete Workflow           ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "워크플로우 단계:"
    echo "  1) 환경 확인"
    echo "  2) 시각화 생성 (Weight schedules)"
    echo "  3) 빠른 기능 테스트"
    echo "  4) Preset 실험 (4가지)"
    echo "  5) 비교 실험 (Temperature, Strategy, Scale)"
    echo "  6) 결과 분석 및 리포트"
    echo ""
    echo "실행 옵션:"
    echo "  ${GREEN}quick${NC}     - 빠른 테스트만 (Step 1-3)"
    echo "  ${GREEN}presets${NC}   - Preset 실험 (Step 1-4)"
    echo "  ${GREEN}full${NC}      - 전체 워크플로우 (Step 1-6)"
    echo "  ${GREEN}analyze${NC}   - 결과 분석만 (Step 6)"
    echo "  ${GREEN}custom${NC}    - 단계별 선택 실행"
    echo "  ${GREEN}help${NC}      - 도움말"
    echo ""
}

# ============================================================================
# 워크플로우 실행
# ============================================================================

workflow_quick() {
    log_section "🚀 Quick Workflow 시작"

    check_environment
    check_classifier
    step_visualize
    step_quick_test

    log_section "✅ Quick Workflow 완료!"
    log_info "결과 확인:"
    log_info "  - 시각화: visualizations/"
    log_info "  - 테스트 이미지: outputs/quick_test/"
}

workflow_presets() {
    log_section "🚀 Preset Workflow 시작"

    check_environment
    check_classifier
    step_visualize
    step_quick_test
    step_presets

    log_section "✅ Preset Workflow 완료!"
    log_info "결과 확인:"
    log_info "  - 시각화: visualizations/"
    log_info "  - Preset 결과: outputs/soft_cg_experiments/"
}

workflow_full() {
    log_section "🚀 Full Workflow 시작"

    check_environment
    check_classifier
    step_visualize
    step_quick_test
    step_presets
    step_comparisons
    step_analyze

    log_section "✅ Full Workflow 완료!"
    log_success "모든 실험이 성공적으로 완료되었습니다!"
    echo ""
    log_info "결과 확인:"
    log_info "  📊 리포트: EXPERIMENT_REPORT.md"
    log_info "  📈 통계: experiment_statistics.json"
    log_info "  🖼️  시각화: visualizations/"
    log_info "  📁 실험 결과: outputs/soft_cg_experiments/"
    echo ""
    log_info "다음 단계:"
    log_info "  1. EXPERIMENT_REPORT.md 확인"
    log_info "  2. visualizations/ 폴더의 그래프 확인"
    log_info "  3. 최적의 preset 선택"
    log_info "  4. 프로덕션 환경에 적용"
}

workflow_analyze() {
    log_section "🚀 Analysis Workflow 시작"

    step_analyze

    log_section "✅ Analysis 완료!"
}

workflow_custom() {
    log_section "🎯 Custom Workflow"

    check_environment

    echo ""
    echo "실행할 단계를 선택하세요 (공백으로 구분):"
    echo "  1: 시각화"
    echo "  2: 빠른 테스트"
    echo "  3: Preset 실험"
    echo "  4: 비교 실험"
    echo "  5: 결과 분석"
    echo ""
    echo "예시: 1 3 5"
    read -r STEPS

    for step in $STEPS; do
        case $step in
            1)
                step_visualize
                ;;
            2)
                check_classifier
                step_quick_test
                ;;
            3)
                check_classifier
                step_presets
                ;;
            4)
                check_classifier
                step_comparisons
                ;;
            5)
                step_analyze
                ;;
            *)
                log_warning "알 수 없는 단계: $step"
                ;;
        esac
    done

    log_section "✅ Custom Workflow 완료!"
}

# ============================================================================
# 사용법
# ============================================================================

show_usage() {
    cat << EOF
Complete Workflow Script

사용법:
  $0 [OPTION]

옵션:
  quick         빠른 테스트 (시각화 + 기본 테스트)
  presets       Preset 실험 (시각화 + 테스트 + Preset 4개)
  full          전체 워크플로우 (모든 단계)
  analyze       결과 분석만
  custom        단계별 선택 실행
  menu          대화형 메뉴
  help          이 도움말

환경 변수:
  CLASSIFIER_PATH   Classifier 체크포인트 경로
                    (기본값: checkpoints/nude_classifier_best.pth)

예시:
  $0 quick                              # 빠른 테스트
  $0 full                               # 전체 워크플로우
  CLASSIFIER_PATH=/path/to/model $0 full  # 커스텀 classifier 경로

단계별 실행:
  1. quick으로 먼저 테스트
  2. 결과 확인 후 presets 실행
  3. 최종적으로 full 실행하여 완전한 분석
EOF
}

# ============================================================================
# 메인
# ============================================================================

main() {
    # 로그 파일 설정
    LOG_FILE="workflow_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1

    case "${1:-menu}" in
        quick)
            workflow_quick
            ;;
        presets)
            workflow_presets
            ;;
        full)
            workflow_full
            ;;
        analyze)
            workflow_analyze
            ;;
        custom)
            workflow_custom
            ;;
        menu)
            show_menu
            echo "실행할 워크플로우를 선택하세요:"
            read -r choice
            case $choice in
                quick|presets|full|analyze|custom)
                    main $choice
                    ;;
                help)
                    show_usage
                    ;;
                *)
                    log_error "알 수 없는 선택: $choice"
                    show_usage
                    exit 1
                    ;;
            esac
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    log_info "로그 저장 위치: $LOG_FILE"
}

# 스크립트 시작
main "$@"
