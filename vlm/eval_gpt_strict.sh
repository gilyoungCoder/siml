#!/bin/bash
# ============================================================================
# VLM Evaluation using STRICT GPT-4o prompts (gpt_i2p_strict.py)
# Lower threshold = More images classified as Partial/Full
#
# Output files: categories_gpt4o_strict_{concept}.json, results_gpt4o_strict_{concept}.txt
# ============================================================================
#
# Usage:
#   ./vlm/eval_gpt_strict.sh [TARGET] [CONCEPT]
#   ./vlm/eval_gpt_strict.sh all all           # All targets, all concepts
#   ./vlm/eval_gpt_strict.sh baselines all     # Only baselines_i2p
#   ./vlm/eval_gpt_strict.sh best_configs all  # Only best_configs
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
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

TARGET=${1:-all}
CONCEPT=${2:-all}

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}VLM Evaluation - STRICT GPT-4o (Lower threshold)${NC}"
echo -e "${GREEN}Target: ${TARGET} | Concept: ${CONCEPT}${NC}"
echo -e "${GREEN}Output: categories_gpt4o_strict_*.json, results_gpt4o_strict_*.txt${NC}"
echo -e "${GREEN}============================================================${NC}"

# VLM concept mapping
declare -A VLM_CONCEPTS
VLM_CONCEPTS["nudity"]="nudity"
VLM_CONCEPTS["violence"]="violence"
VLM_CONCEPTS["harassment"]="harassment"
VLM_CONCEPTS["hate"]="hate"
VLM_CONCEPTS["shocking"]="shocking"
VLM_CONCEPTS["illegal"]="illegal"
VLM_CONCEPTS["selfharm"]="self_harm"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

# Class type mapping
declare -A CLASS_TYPES
CLASS_TYPES["nudity"]="4class"
CLASS_TYPES["violence"]="13class"
CLASS_TYPES["harassment"]="9class"
CLASS_TYPES["hate"]="9class"
CLASS_TYPES["shocking"]="9class"
CLASS_TYPES["illegal"]="9class"
CLASS_TYPES["selfharm"]="9class"

run_eval() {
    local img_dir=$1
    local vlm_concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}[SKIP] Directory not found: ${img_dir}${NC}"
        return
    fi

    result_file="$img_dir/results_gpt4o_strict_${vlm_concept}.txt"

    if [ -f "$result_file" ]; then
        echo -e "${YELLOW}[SKIP] Already evaluated: ${desc}${NC}"
        return
    fi

    echo -e "\n${CYAN}[${desc}]${NC}"
    echo "Directory: ${img_dir}"

    python vlm/gpt_i2p_strict.py "$img_dir" "$vlm_concept"
}

# Baselines evaluation
eval_baselines() {
    local concepts_to_eval=("$@")

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  BASELINES_I2P Evaluation (STRICT)${NC}"
    echo -e "${GREEN}========================================${NC}"

    for method in "sd_baseline" "safree"; do
        echo -e "\n${GREEN}--- Method: ${method} ---${NC}"

        for concept in "${concepts_to_eval[@]}"; do
            local vlm_concept="${VLM_CONCEPTS[$concept]}"

            for tox in "high_tox" "low_tox"; do
                local img_dir="${BASELINES_DIR}/${method}/${concept}/${tox}"
                run_eval "$img_dir" "$vlm_concept" "${method}/${concept}/${tox}"
            done
        done
    done
}

# Best configs evaluation
eval_best_configs() {
    local concepts_to_eval=("$@")

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  BEST_CONFIGS Evaluation (STRICT)${NC}"
    echo -e "${GREEN}========================================${NC}"

    for concept in "${concepts_to_eval[@]}"; do
        local vlm_concept="${VLM_CONCEPTS[$concept]}"
        local class_type="${CLASS_TYPES[$concept]}"

        echo -e "\n${GREEN}--- Concept: ${concept} ---${NC}"

        # Ours (skip_ca)
        local folder="${concept}_${class_type}_skip_ca"
        for tox in "high_tox" "low_tox"; do
            local img_dir="${BEST_CONFIGS_DIR}/${folder}/${tox}"
            run_eval "$img_dir" "$vlm_concept" "best_configs/${folder}/${tox}"
        done

        # SAFREE + Ours
        local safree_folder="safree_ours_${concept}_${class_type}"
        for tox in "high_tox" "low_tox"; do
            local img_dir="${BEST_CONFIGS_DIR}/${safree_folder}/${tox}"
            run_eval "$img_dir" "$vlm_concept" "best_configs/${safree_folder}/${tox}"
        done
    done
}

# Determine concepts
if [ "$CONCEPT" == "all" ]; then
    CONCEPTS_TO_EVAL=("${ALL_CONCEPTS[@]}")
else
    if [[ ! -v "VLM_CONCEPTS[$CONCEPT]" ]]; then
        echo -e "${RED}Unknown concept: ${CONCEPT}${NC}"
        exit 1
    fi
    CONCEPTS_TO_EVAL=("$CONCEPT")
fi

# Run evaluation
case $TARGET in
    "baselines")
        eval_baselines "${CONCEPTS_TO_EVAL[@]}"
        ;;
    "best_configs")
        eval_best_configs "${CONCEPTS_TO_EVAL[@]}"
        ;;
    "all")
        eval_baselines "${CONCEPTS_TO_EVAL[@]}"
        eval_best_configs "${CONCEPTS_TO_EVAL[@]}"
        ;;
    *)
        echo -e "${RED}Unknown target: ${TARGET}${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}STRICT GPT-4o Evaluation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
