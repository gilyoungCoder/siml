#!/bin/bash
# ============================================================================
# VLM Evaluation for Baselines (SD Baseline + SAFREE)
# ============================================================================
#
# Usage:
#   ./vlm/eval_baselines_i2p.sh [CONCEPT] [METHOD] [GPU]
#   ./vlm/eval_baselines_i2p.sh all all 0         # All concepts, all methods
#   ./vlm/eval_baselines_i2p.sh nudity sd 0       # SD baseline for nudity
#   ./vlm/eval_baselines_i2p.sh violence safree 1 # SAFREE for violence
#
# ============================================================================

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BASELINES_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"

CONCEPT=${1:-all}
METHOD=${2:-all}  # sd, safree, or all
GPU=${3:-0}

export CUDA_VISIBLE_DEVICES=$GPU

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}VLM Evaluation - Baselines (GPT-4o)${NC}"
echo -e "${GREEN}Concept: ${CONCEPT} | Method: ${METHOD} | GPU: ${GPU}${NC}"
echo -e "${GREEN}============================================================${NC}"

# VLM concept mapping (folder name -> vlm concept name)
declare -A VLM_CONCEPTS
VLM_CONCEPTS["nudity"]="nudity"
VLM_CONCEPTS["violence"]="violence"
VLM_CONCEPTS["harassment"]="harassment"
VLM_CONCEPTS["hate"]="hate"
VLM_CONCEPTS["shocking"]="shocking"
VLM_CONCEPTS["illegal"]="illegal"
VLM_CONCEPTS["selfharm"]="self_harm"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")
METHODS_SD=("sd_baseline")
METHODS_SAFREE=("safree")

run_eval() {
    local img_dir=$1
    local vlm_concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}[SKIP] Directory not found: ${img_dir}${NC}"
        return
    fi

    result_file="$img_dir/results_gpt4o_${vlm_concept}.txt"

    if [ -f "$result_file" ]; then
        echo -e "${YELLOW}[SKIP] Already evaluated: ${desc}${NC}"
        return
    fi

    echo -e "\n${CYAN}[${desc}]${NC}"
    echo "Directory: ${img_dir}"

    python vlm/gpt_i2p_all.py "$img_dir" "$vlm_concept"
}

eval_method_concept() {
    local method=$1
    local concept=$2
    local vlm_concept="${VLM_CONCEPTS[$concept]}"

    for tox in "high_tox" "low_tox"; do
        local img_dir="${BASELINES_DIR}/${method}/${concept}/${tox}"
        run_eval "$img_dir" "$vlm_concept" "${method} ${concept} (${tox})"
    done
}

# Determine which methods to run
if [ "$METHOD" == "all" ]; then
    METHODS=("sd_baseline" "safree")
elif [ "$METHOD" == "sd" ]; then
    METHODS=("sd_baseline")
elif [ "$METHOD" == "safree" ]; then
    METHODS=("safree")
else
    echo -e "${RED}Unknown method: ${METHOD}${NC}"
    echo "Available: sd, safree, all"
    exit 1
fi

# Determine which concepts to run
if [ "$CONCEPT" == "all" ]; then
    CONCEPTS=("${ALL_CONCEPTS[@]}")
else
    if [[ ! -v "VLM_CONCEPTS[$CONCEPT]" ]]; then
        echo -e "${RED}Unknown concept: ${CONCEPT}${NC}"
        echo "Available: nudity, violence, harassment, hate, shocking, illegal, selfharm, all"
        exit 1
    fi
    CONCEPTS=("$CONCEPT")
fi

# Run evaluation
for method in "${METHODS[@]}"; do
    echo -e "\n${GREEN}========== ${method^^} ==========${NC}"
    for concept in "${CONCEPTS[@]}"; do
        eval_method_concept "$method" "$concept"
    done
done

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Baselines Evaluation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
