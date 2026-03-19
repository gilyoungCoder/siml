#!/bin/bash
# ============================================================================
# Parallel Launcher for I2P Baselines (SAFREE + SD Baseline)
# ============================================================================
#
# Usage:
#   ./scripts/generate_i2p_baselines_parallel.sh <METHOD> [START_GPU]
#   ./scripts/generate_i2p_baselines_parallel.sh baseline 0   # SD baseline GPU 0-6
#   ./scripts/generate_i2p_baselines_parallel.sh safree 0     # SAFREE GPU 0-6
#
# ============================================================================

METHOD=${1:-baseline}
START_GPU=${2:-0}

cd /mnt/home/yhgil99/unlearning
mkdir -p logs

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

echo -e "${GREEN}=============================================="
echo -e "Launching I2P Baselines in PARALLEL"
echo -e "Method: ${METHOD}"
echo -e "==============================================${NC}"
echo ""

for i in "${!CONCEPTS[@]}"; do
    CONCEPT="${CONCEPTS[$i]}"
    GPU=$((START_GPU + i))
    
    echo -e "${YELLOW}Launching: ${CONCEPT} (${METHOD}) on GPU ${GPU}${NC}"
    
    nohup bash scripts/generate_i2p_baselines.sh "$METHOD" "$GPU" "$CONCEPT" \
        > "logs/baseline_${METHOD}_${CONCEPT}.log" 2>&1 &
    
    echo "  PID: $!"
    echo "  Log: logs/baseline_${METHOD}_${CONCEPT}.log"
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
    echo "  GPU $GPU: $CONCEPT ($METHOD)"
done
echo ""
echo "Monitor: tail -f logs/baseline_${METHOD}_*.log"
