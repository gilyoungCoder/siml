#!/bin/bash
# ============================================================================
# Re-evaluate Baselines with Updated VLM Prompt
#
# Usage:
#   ./vlm/eval_baselines.sh <GPU>
#   ./vlm/eval_baselines.sh 0
# ============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <GPU>"
    echo "Example: $0 0"
    exit 1
fi

GPU="$1"
CONCEPT="nudity"

export CUDA_VISIBLE_DEVICES=${GPU}

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"
BASELINES_DIR="SoftDelete+CG/scg_outputs/baselines_i2p_sexual"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}=============================================="
echo -e "Re-evaluate Baselines with Updated VLM Prompt"
echo -e "==============================================${NC}"
echo "GPU: ${GPU}"
echo "Concept: ${CONCEPT}"
echo ""

# List of baselines to evaluate
BASELINES=(
    "sd_baseline"
    "safree"
)

for baseline in "${BASELINES[@]}"; do
    BASELINE_DIR="${BASELINES_DIR}/${baseline}"

    if [ ! -d "$BASELINE_DIR" ]; then
        echo -e "${YELLOW}[SKIP] ${baseline} - directory not found${NC}"
        continue
    fi

    # Check if images exist
    IMG_COUNT=$(ls "$BASELINE_DIR"/*.png 2>/dev/null | wc -l)
    if [ "$IMG_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}[SKIP] ${baseline} - no images found${NC}"
        continue
    fi

    # Delete old evaluation results
    echo -e "${YELLOW}[DELETE] Removing old results in ${baseline}${NC}"
    rm -f "$BASELINE_DIR"/results_*.txt
    rm -f "$BASELINE_DIR"/categories_*.json

    echo -e "${CYAN}[EVAL] ${baseline} (${IMG_COUNT} images)${NC}"
    python "$VLM_SCRIPT" "$BASELINE_DIR" "$CONCEPT" qwen
    echo ""
done

echo -e "${GREEN}=============================================="
echo -e "Evaluation Complete!"
echo -e "==============================================${NC}"

# Summary
echo ""
echo "Results Summary:"
echo "----------------"

for baseline in "${BASELINES[@]}"; do
    BASELINE_DIR="${BASELINES_DIR}/${baseline}"
    RESULTS_FILE="${BASELINE_DIR}/results_qwen2_vl_${CONCEPT}.txt"

    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo -e "${CYAN}=== ${baseline} ===${NC}"
        cat "$RESULTS_FILE"
    fi
done

echo ""
echo -e "${GREEN}Done!${NC}"
