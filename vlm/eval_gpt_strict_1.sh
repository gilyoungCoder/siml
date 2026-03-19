#!/bin/bash
# ============================================================================
# STRICT GPT-4o Evaluation - Part 1: BASELINES_I2P
# ============================================================================

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BASELINES_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"

CONCEPT=${1:-all}

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}STRICT Evaluation - Part 1: BASELINES_I2P${NC}"
echo -e "${GREEN}Concept: ${CONCEPT}${NC}"
echo -e "${GREEN}============================================================${NC}"

declare -A VLM_CONCEPTS
VLM_CONCEPTS["nudity"]="nudity"
VLM_CONCEPTS["violence"]="violence"
VLM_CONCEPTS["harassment"]="harassment"
VLM_CONCEPTS["hate"]="hate"
VLM_CONCEPTS["shocking"]="shocking"
VLM_CONCEPTS["illegal"]="illegal"
VLM_CONCEPTS["selfharm"]="self_harm"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

run_eval() {
    local img_dir=$1
    local vlm_concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}[SKIP] Not found: ${img_dir}${NC}"
        return
    fi

    result_file="$img_dir/results_gpt4o_strict_${vlm_concept}.txt"
    if [ -f "$result_file" ]; then
        echo -e "${YELLOW}[SKIP] Already done: ${desc}${NC}"
        return
    fi

    echo -e "\n${CYAN}[${desc}]${NC}"
    python vlm/gpt_i2p_strict.py "$img_dir" "$vlm_concept"
}

if [ "$CONCEPT" == "all" ]; then
    CONCEPTS=("${ALL_CONCEPTS[@]}")
else
    CONCEPTS=("$CONCEPT")
fi

for method in "sd_baseline" "safree"; do
    echo -e "\n${GREEN}--- ${method} ---${NC}"
    for concept in "${CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        for tox in "high_tox" "low_tox"; do
            run_eval "${BASELINES_DIR}/${method}/${concept}/${tox}" "$vlm_concept" "${method}/${concept}/${tox}"
        done
    done
done

echo -e "\n${GREEN}Part 1 (BASELINES) Complete!${NC}"
