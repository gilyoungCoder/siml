#!/bin/bash
# ============================================================================
# Grid Search for ALL I2P Concepts - Both Skip and No-Skip Versions
# ============================================================================
#
# Total: 6 concepts × 2 modes = 12 experiments
# Each GPU runs BOTH modes SIMULTANEOUSLY (2 processes per GPU)
#
# Best checkpoint steps per concept:
#   - Violence: 15500
#   - Shocking: 23700
#   - Illegal: 22600
#   - Self-harm: 20700
#   - Harassment: 24300
#   - Hate: 20800
#
# Usage:
#   ./grid_search_i2p_all.sh <START_GPU>
#   ./grid_search_i2p_all.sh 0      # Uses GPU 0-5 for 6 concepts
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <START_GPU>"
    echo "Example: $0 0      # Uses GPU 0-5 (2 processes per GPU)"
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
CYAN='\033[0;36m'
NC='\033[0m'

# Create logs directory
mkdir -p logs

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Running Grid Search for ALL I2P Concepts (12 Experiments)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "${CYAN}6 GPUs, each runs 2 processes (no-skip + skip) simultaneously${NC}"
echo ""

# Launch each concept on different GPU (both modes run in parallel on same GPU)
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}

    echo -e "${YELLOW}Launching: ${CONCEPT} (step ${STEP}) on GPU ${GPU}${NC}"
    echo "  Running: no-skip + skip (parallel)"

    # Launch no-skip mode
    nohup bash grid_search_i2p_9class_spatial_cg.sh "$CONCEPT" "$STEP" "$GPU" "" \
        > "logs/grid_search_${CONCEPT}_noskip.log" 2>&1 &
    PID1=$!
    echo "  [no-skip] PID: $PID1, Log: logs/grid_search_${CONCEPT}_noskip.log"

    # Launch skip mode (same GPU)
    nohup bash grid_search_i2p_9class_spatial_cg.sh "$CONCEPT" "$STEP" "$GPU" "skip" \
        > "logs/grid_search_${CONCEPT}_skip.log" 2>&1 &
    PID2=$!
    echo "  [skip]    PID: $PID2, Log: logs/grid_search_${CONCEPT}_skip.log"

    echo ""
done

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All 12 processes launched (2 per GPU)!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "GPU Assignments:"
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}
    echo "  GPU $GPU: $CONCEPT (step $STEP) [no-skip + skip parallel]"
done
echo ""
echo "Monitor with:"
echo "  tail -f logs/grid_search_*_noskip.log"
echo "  tail -f logs/grid_search_*_skip.log"
echo ""
echo "Check status: ps aux | grep grid_search"
