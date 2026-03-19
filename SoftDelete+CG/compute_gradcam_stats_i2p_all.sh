#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for ALL I2P Concepts (using best checkpoints)
# ============================================================================
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
#   ./compute_gradcam_stats_i2p_all.sh <START_GPU>
#   ./compute_gradcam_stats_i2p_all.sh 0      # Uses GPU 0,1,2,3,4,5
#   ./compute_gradcam_stats_i2p_all.sh 2      # Uses GPU 2,3,4,5,6,7
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <START_GPU>"
    echo "Example: $0 0      # Uses GPU 0,1,2,3,4,5 for 6 concepts"
    echo "Example: $0 2      # Uses GPU 2,3,4,5,6,7 for 6 concepts"
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

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Computing GradCAM Statistics for ALL I2P Concepts (Parallel)${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Launch each concept on different GPU
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}

    echo -e "${YELLOW}Launching: ${CONCEPT} (step ${STEP}) on GPU ${GPU}${NC}"

    # Run in background with nohup
    nohup bash compute_gradcam_stats_i2p_9class.sh "$CONCEPT" "$STEP" "$GPU" \
        > "logs/gradcam_stats_${CONCEPT}.log" 2>&1 &

    echo "  PID: $!"
    echo "  Log: logs/gradcam_stats_${CONCEPT}.log"
    echo ""
done

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All processes launched!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "GPU Assignments:"
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    STEP=${BEST_STEPS[$CONCEPT]}
    echo "  GPU $GPU: $CONCEPT (step $STEP)"
done
echo ""
echo "Monitor with: tail -f logs/gradcam_stats_*.log"
echo "Check status: ps aux | grep compute_gradcam"
