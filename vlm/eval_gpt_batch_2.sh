#!/bin/bash
# GPT-4o VLM Batch Evaluation Script - Part 2
# Concepts: hate, shocking, illegal, selfharm
# Run in parallel with eval_gpt_batch_1.sh
#
# Usage: ./vlm/eval_gpt_batch_2.sh

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
SAFREE_BASELINE="${BASE_DIR}/SAFREE/safree_outputs/i2p"
SD_BASELINE="${BASE_DIR}/SoftDelete+CG/SDbaseline/i2p"

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o VLM Batch Evaluation - Part 2${NC}"
echo -e "${GREEN}Concepts: hate, shocking, illegal, selfharm${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval() {
    local img_dir=$1
    local concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}Skipping (not found): ${img_dir}${NC}"
        return
    fi

    if [ -f "$img_dir/results_gpt4o_${concept}.txt" ]; then
        echo -e "${YELLOW}[SKIP] Already evaluated: ${desc}${NC}"
        return
    fi

    echo -e "\n${GREEN}[${desc}]${NC}"
    echo "Directory: ${img_dir}"
    echo "Concept: ${concept}"

    python vlm/gpt_i2p_all.py "$img_dir" "$concept"
}

# ============================================================================
# HATE Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}HATE EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/hate" "hate" "SAFREE Baseline"
run_eval "${SD_BASELINE}/hate" "hate" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs2.0_bgs1.0_cos" "hate" "hate_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs1.0_bgs2.0_lin" "hate" "hate_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.4-0.2_hs1.0_bgs1.0_lin" "hate" "hate_9class #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs12.5_st0.5-0.5_hs1.5_bgs2.0_cos" "hate" "hate_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos" "hate" "hate_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs7.5_st0.2-0.4_hs2.0_bgs2.0_cos" "hate" "hate_9class_skip #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs7.5_st0.5-0.5_hs1.5_bgs0.0" "hate" "SAFREE_hate #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs5.0_st0.7-0.3_hs1.5_bgs1.0" "hate" "SAFREE_hate #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs5.0_st0.5-0.5_hs1.5_bgs1.0" "hate" "SAFREE_hate #3"

# ============================================================================
# SHOCKING Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}SHOCKING EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/shocking" "shocking" "SAFREE Baseline"
run_eval "${SD_BASELINE}/shocking" "shocking" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "shocking" "shocking_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" "shocking" "shocking_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" "shocking" "shocking_9class_skip #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_cos" "shocking" "shocking_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_lin" "shocking" "shocking_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.4-0.6_hs1.0_bgs1.0_cos" "shocking" "shocking_9class #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.7_hs1.0_bgs2.0" "shocking" "SAFREE_shocking #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.5_hs1.5_bgs0.0" "shocking" "SAFREE_shocking #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs5.0_st0.7-0.3_hs1.0_bgs2.0" "shocking" "SAFREE_shocking #3"

# ============================================================================
# ILLEGAL Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}ILLEGAL EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/illegal activity" "illegal" "SAFREE Baseline"
run_eval "${SD_BASELINE}/illegal_activity" "illegal" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.7_hs1.5_bgs2.0_cos" "illegal" "illegal_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs12.5_st0.2-0.4_hs1.0_bgs2.0_cos" "illegal" "illegal_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.5_hs1.5_bgs2.0_cos" "illegal" "illegal_9class_skip #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.0_bgs2.0_lin" "illegal" "illegal_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" "illegal" "illegal_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs10.0_st0.5-0.5_hs1.5_bgs1.0_lin" "illegal" "illegal_9class #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs5.0_st0.5-0.3_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs5.0_st0.5-0.5_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs10.0_st0.5-0.5_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #3"

# ============================================================================
# SELF-HARM Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}SELF-HARM EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/self-harm" "self_harm" "SAFREE Baseline"
run_eval "${SD_BASELINE}/selfharm" "self_harm" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" "self_harm" "selfharm_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.0_bgs1.0_lin" "self_harm" "selfharm_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.5_bgs1.0_cos" "self_harm" "selfharm_9class #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.5-0.3_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.5-0.5_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs0.5_bgs0.0" "self_harm" "SAFREE_selfharm #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs1.0_bgs0.0" "self_harm" "SAFREE_selfharm #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs1.0_bgs1.0" "self_harm" "SAFREE_selfharm #3"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o VLM Batch Evaluation Part 2 Complete${NC}"
echo -e "${GREEN}============================================================${NC}"
