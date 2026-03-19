#!/bin/bash
# ============================================================================
# Parallel Launcher for Best Config Generation (ALL Concepts)
# ============================================================================
#
# Usage:
#   ./generate_best_configs_parallel.sh [START_GPU]
#   ./generate_best_configs_parallel.sh 0     # Uses GPU 0-6 for 7 concepts
#
# ============================================================================

START_GPU=${1:-0}

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
mkdir -p logs

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

echo -e "${GREEN}=============================================="
echo -e "Launching ALL concepts in PARALLEL (Best Configs)"
echo -e "==============================================${NC}"
echo ""

for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))

    echo -e "${YELLOW}Launching: ${CONCEPT} on GPU ${GPU}${NC}"

    nohup bash generate_best_configs_all.sh "$CONCEPT" "$GPU" both \
        > "logs/generate_best_${CONCEPT}.log" 2>&1 &

    echo "  PID: $!"
    echo "  Log: logs/generate_best_${CONCEPT}.log"
    echo ""
done

echo -e "${GREEN}=============================================="
echo -e "All 7 concepts launched!"
echo -e "==============================================${NC}"
echo ""
echo "GPU Assignments:"
for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    echo "  GPU $GPU: $CONCEPT"
done
echo ""
echo "Monitor: tail -f logs/generate_best_*.log"
echo "Check progress: ps aux | grep generate"
