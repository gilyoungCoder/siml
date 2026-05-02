#!/bin/bash
# ============================================================================
# Grid Search for ALL I2P Concepts - SKIP Version Only
# ============================================================================
#
# Total: 6 concepts (skip mode only)
# Each GPU runs 1 process
#
# Usage:
#   ./grid_search_i2p_all_skip.sh <START_GPU>
#   ./grid_search_i2p_all_skip.sh 0      # Uses GPU 0-5
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <START_GPU>"
    echo "Example: $0 0      # Uses GPU 0-5 for 6 concepts (skip mode)"
    exit 1
fi

START_GPU=$1

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Best steps per concept
declare -A BEST_STEPS
BEST_STEPS["violence"]=15500
BEST_STEPS["shocking"]=23700
BEST_STEPS["illegal"]=22600
BEST_STEPS["selfharm"]=20700
BEST_STEPS["harassment"]=24300
BEST_STEPS["hate"]=20800

# Concept list
CONCEPTS=("violence" "shocking" "illegal" "selfharm" "harassment" "hate")

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create logs directory
mkdir -p logs

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Running Grid Search - SKIP Mode (6 Concepts)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}

    echo -e "${YELLOW}Launching: ${CONCEPT} [skip] (step ${STEP}) on GPU ${GPU}${NC}"

    nohup bash grid_search_i2p_9class_spatial_cg.sh "$CONCEPT" "$STEP" "$GPU" "skip" \
        > "logs/grid_search_${CONCEPT}_skip.log" 2>&1 &

    echo "  PID: $!"
    echo "  Log: logs/grid_search_${CONCEPT}_skip.log"
    echo ""
done

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All 6 SKIP processes launched!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "GPU Assignments:"
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}
    echo "  GPU $GPU: $CONCEPT (step $STEP) [skip]"
done
echo ""
echo "Monitor: tail -f logs/grid_search_*_skip.log"
