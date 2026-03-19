#!/bin/bash
# ============================================================================
# VLM Evaluation for Best Configs (high_tox + low_tox)
# ============================================================================
#
# Usage:
#   ./vlm/eval_best_configs.sh [CONCEPT] [MODEL]
#   ./vlm/eval_best_configs.sh nudity gpt       # GPT-4o for nudity
#   ./vlm/eval_best_configs.sh all qwen         # Qwen for all concepts
#   ./vlm/eval_best_configs.sh all gpt          # GPT-4o for all concepts
#
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

CONCEPT=${1:-all}
MODEL=${2:-qwen}  # qwen or gpt

# Validate model
if [[ "$MODEL" != "qwen" && "$MODEL" != "gpt" ]]; then
    echo -e "${RED}Error: MODEL must be 'qwen' or 'gpt'${NC}"
    exit 1
fi

# Check API key for GPT
if [ "$MODEL" == "gpt" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}VLM Evaluation - Best Configs${NC}"
echo -e "${GREEN}Model: ${MODEL}${NC}"
echo -e "${GREEN}Concept: ${CONCEPT}${NC}"
echo -e "${GREEN}============================================================${NC}"

# Config mapping: concept -> class_type|vlm_concept_name
declare -A CONFIGS
CONFIGS["nudity"]="4class|nudity"
CONFIGS["violence"]="13class|violence"
CONFIGS["harassment"]="9class|harassment"
CONFIGS["hate"]="9class|hate"
CONFIGS["shocking"]="9class|shocking"
CONFIGS["illegal"]="9class|illegal"
CONFIGS["selfharm"]="9class|self_harm"

# Variants to evaluate: skip (linear) and skip_ca (cosine anneal)
VARIANTS=("skip" "skip_ca")

# Function to run evaluation
run_eval() {
    local img_dir=$1
    local concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}[SKIP] Directory not found: ${img_dir}${NC}"
        return
    fi

    # Check if already evaluated
    if [ "$MODEL" == "gpt" ]; then
        result_file="$img_dir/results_gpt4o_${concept}.txt"
    else
        result_file="$img_dir/results_qwen2vl_${concept}.txt"
    fi

    if [ -f "$result_file" ]; then
        echo -e "${YELLOW}[SKIP] Already evaluated: ${desc}${NC}"
        return
    fi

    echo -e "\n${CYAN}[${desc}]${NC}"
    echo "Directory: ${img_dir}"
    echo "Concept: ${concept}"

    if [ "$MODEL" == "gpt" ]; then
        python vlm/gpt_i2p_all.py "$img_dir" "$concept"
    else
        python vlm/opensource_vlm_i2p_all.py "$img_dir" "$concept" qwen
    fi
}

# Function to evaluate a single concept
eval_concept() {
    local concept=$1

    IFS='|' read -r class_type vlm_concept <<< "${CONFIGS[$concept]}"

    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}Evaluating: ${concept^^}${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Evaluate both variants (skip and skip_ca)
    for variant in "${VARIANTS[@]}"; do
        local folder_name="${concept}_${class_type}_${variant}"

        # High toxicity
        run_eval "${BEST_CONFIGS_DIR}/${folder_name}/high_tox" "$vlm_concept" "${concept} ${variant} (high_tox)"

        # Low toxicity
        run_eval "${BEST_CONFIGS_DIR}/${folder_name}/low_tox" "$vlm_concept" "${concept} ${variant} (low_tox)"
    done
}

# Main execution
ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

if [ "$CONCEPT" == "all" ]; then
    for c in "${ALL_CONCEPTS[@]}"; do
        eval_concept "$c"
    done
else
    if [[ ! -v "CONFIGS[$CONCEPT]" ]]; then
        echo -e "${RED}Unknown concept: ${CONCEPT}${NC}"
        echo "Available: nudity, violence, harassment, hate, shocking, illegal, selfharm, all"
        exit 1
    fi
    eval_concept "$CONCEPT"
fi

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
