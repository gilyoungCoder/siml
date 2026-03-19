#!/bin/bash
# ============================================================================
# Run All Grid Search Experiments with nohup
# ============================================================================

# 사용법:
#   ./run_grid_search_all.sh [nudity|vangogh|violence|all]
#
# 예시:
#   ./run_grid_search_all.sh nudity      # nudity만 실행
#   ./run_grid_search_all.sh vangogh     # vangogh만 실행
#   ./run_grid_search_all.sh violence    # violence만 실행
#   ./run_grid_search_all.sh all         # 전부 순차 실행

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TARGET=${1:-all}

run_nudity() {
    echo -e "${CYAN}[Nudity]${NC} Starting grid search..."
    nohup ./grid_search_spatial_threshold.sh > grid_search_nudity.log 2>&1 &
    echo -e "${GREEN}[Nudity]${NC} PID: $! | Log: grid_search_nudity.log"
}

run_vangogh() {
    echo -e "${CYAN}[VanGogh]${NC} Starting grid search..."
    nohup ./grid_search_spatial_threshold_vangogh.sh > grid_search_vangogh.log 2>&1 &
    echo -e "${GREEN}[VanGogh]${NC} PID: $! | Log: grid_search_vangogh.log"
}

run_violence() {
    echo -e "${CYAN}[Violence]${NC} Starting grid search..."
    nohup ./grid_search_spatial_threshold_violence.sh > grid_search_violence.log 2>&1 &
    echo -e "${GREEN}[Violence]${NC} PID: $! | Log: grid_search_violence.log"
}

case $TARGET in
    nudity)
        run_nudity
        ;;
    vangogh)
        run_vangogh
        ;;
    violence)
        run_violence
        ;;
    all)
        echo -e "${YELLOW}Running all grid searches sequentially...${NC}"
        echo ""
        run_nudity
        run_vangogh
        run_violence
        ;;
    *)
        echo "Usage: $0 [nudity|vangogh|violence|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Grid search started!${NC}"
echo "Check progress with: tail -f grid_search_*.log"
