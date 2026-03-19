#!/bin/bash
# ============================================================================
# STRICT GPT-4o Evaluation - Part 2: BEST_CONFIGS
# ============================================================================

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

CONCEPT=${1:-all}

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}STRICT Evaluation - Part 2: BEST_CONFIGS${NC}"
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

declare -A CLASS_TYPES
CLASS_TYPES["nudity"]="4class"
CLASS_TYPES["violence"]="13class"
CLASS_TYPES["harassment"]="9class"
CLASS_TYPES["hate"]="9class"
CLASS_TYPES["shocking"]="9class"
CLASS_TYPES["illegal"]="9class"
CLASS_TYPES["selfharm"]="9class"

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

for concept in "${CONCEPTS[@]}"; do
    vlm_concept="${VLM_CONCEPTS[$concept]}"
    class_type="${CLASS_TYPES[$concept]}"

    echo -e "\n${GREEN}--- ${concept} ---${NC}"

    # Ours (skip_ca)
    folder="${concept}_${class_type}_skip_ca"
    for tox in "high_tox" "low_tox"; do
        run_eval "${BEST_CONFIGS_DIR}/${folder}/${tox}" "$vlm_concept" "ours/${folder}/${tox}"
    done

    # SAFREE + Ours
    safree_folder="safree_ours_${concept}_${class_type}"
    for tox in "high_tox" "low_tox"; do
        run_eval "${BEST_CONFIGS_DIR}/${safree_folder}/${tox}" "$vlm_concept" "safree_ours/${safree_folder}/${tox}"
    done
done

echo -e "\n${GREEN}Part 2 (BEST_CONFIGS) Complete!${NC}"
