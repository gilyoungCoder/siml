#!/bin/bash
# ============================================================================
# VLM Evaluation for Best Configs - LOW TOXICITY ONLY (skip + skip_ca)
# ============================================================================

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

CONCEPT=${1:-all}
MODEL=${2:-qwen}
GPU=${3:-1}

export CUDA_VISIBLE_DEVICES=$GPU

if [[ "$MODEL" != "qwen" && "$MODEL" != "gpt" ]]; then
    echo -e "${RED}Error: MODEL must be 'qwen' or 'gpt'${NC}"
    exit 1
fi

if [ "$MODEL" == "gpt" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}VLM Evaluation - Best Configs (LOW_TOX) [skip + skip_ca]${NC}"
echo -e "${GREEN}Model: ${MODEL} | GPU: ${GPU}${NC}"
echo -e "${GREEN}Concept: ${CONCEPT}${NC}"
echo -e "${GREEN}============================================================${NC}"

declare -A CONFIGS
CONFIGS["nudity"]="4class|nudity"
CONFIGS["violence"]="13class|violence"
CONFIGS["harassment"]="9class|harassment"
CONFIGS["hate"]="9class|hate"
CONFIGS["shocking"]="9class|shocking"
CONFIGS["illegal"]="9class|illegal"
CONFIGS["selfharm"]="9class|self_harm"

VARIANTS=("skip" "skip_ca")

run_eval() {
    local img_dir=$1
    local concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}[SKIP] Directory not found: ${img_dir}${NC}"
        return
    fi

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

    if [ "$MODEL" == "gpt" ]; then
        python vlm/gpt_i2p_all.py "$img_dir" "$concept"
    else
        python vlm/opensource_vlm_i2p_all.py "$img_dir" "$concept" qwen
    fi
}

eval_concept() {
    local concept=$1
    IFS='|' read -r class_type vlm_concept <<< "${CONFIGS[$concept]}"

    echo -e "\n${GREEN}Evaluating: ${concept^^} (low_tox)${NC}"

    for variant in "${VARIANTS[@]}"; do
        local folder_name="${concept}_${class_type}_${variant}"
        run_eval "${BEST_CONFIGS_DIR}/${folder_name}/low_tox" "$vlm_concept" "${concept} ${variant} (low_tox)"
    done
}

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

if [ "$CONCEPT" == "all" ]; then
    for c in "${ALL_CONCEPTS[@]}"; do
        eval_concept "$c"
    done
else
    if [[ ! -v "CONFIGS[$CONCEPT]" ]]; then
        echo -e "${RED}Unknown concept: ${CONCEPT}${NC}"
        exit 1
    fi
    eval_concept "$CONCEPT"
fi

echo -e "\n${GREEN}LOW_TOX Evaluation Complete!${NC}"
