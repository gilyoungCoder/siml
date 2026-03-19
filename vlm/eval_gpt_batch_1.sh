#!/bin/bash
# GPT-4o VLM Batch Evaluation Script - Part 1
# Concepts: nudity, violence, harassment
# Run in parallel with eval_gpt_batch_2.sh
#
# Usage: ./vlm/eval_gpt_batch_1.sh

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
echo -e "${GREEN}GPT-4o VLM Batch Evaluation - Part 1${NC}"
echo -e "${GREEN}Concepts: nudity, violence, harassment${NC}"
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
# NUDITY (sexual) Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}NUDITY/SEXUAL EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/sexual" "nudity" "SAFREE Baseline"
run_eval "${SD_BASELINE}/sexual" "nudity" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.5-0.3_hs1.5_bgs2.0_cosine_anneal" "nudity" "nudity_4class_always #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.5-0.1_hs1.0_bgs2.0_cosine_anneal" "nudity" "nudity_4class_always #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.3-0.5_hs1.5_bgs1.0_cosine_anneal" "nudity" "nudity_4class_always #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.3-0.3_hs0.5_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2 #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.3-0.3_hs1.5_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2 #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.7-0.5_hs0.5_bgs2.0_linear_decrease" "nudity" "nudity_6class_v2 #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.3_hs1.0_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.3_hs1.5_bgs0.0_linear_decrease" "nudity" "nudity_6class_v2_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.5_hs0.5_bgs0.0_linear_decrease" "nudity" "nudity_6class_v2_skip #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs5.0_st0.5-0.3_hs0.5_bgs2.0" "nudity" "SAFREE_nudity #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs5.0_st0.5-0.5_hs0.5_bgs2.0" "nudity" "SAFREE_nudity #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs7.5_st0.7-0.3_hs1.5_bgs0.0" "nudity" "SAFREE_nudity #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal" "nudity" "nudity_4class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs10.0_thr0.7-0.3_hs0.5_bgs1.0_cosine_anneal" "nudity" "nudity_4class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs10.0_thr0.7-0.3_hs1.5_bgs2.0_linear_decrease" "nudity" "nudity_4class #3"

# ============================================================================
# VIOLENCE Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}VIOLENCE EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/violence" "violence" "SAFREE Baseline"
run_eval "${SD_BASELINE}/violence" "violence" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs12.5_st0.4-0.6_hs1.0_bgs1.0_lin" "violence" "violence_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" "violence" "violence_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs12.5_st0.4-0.6_hs1.0_bgs1.0_cos" "violence" "violence_9class_skip #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "violence" "violence_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" "violence" "violence_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" "violence" "violence_9class #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs5.0_st0.7-0.3_hs1.0_bgs0.0" "violence" "SAFREE_violence #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs5.0_st0.3-0.7_hs0.5_bgs1.0" "violence" "SAFREE_violence #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs10.0_st0.5-0.3_hs1.0_bgs0.0" "violence" "SAFREE_violence #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.3-0.7_hs1.0_bgs1.0_cosine_anneal" "violence" "violence_13class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.5-0.3_hs1.0_bgs2.0_cosine_anneal" "violence" "violence_13class #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs7.5_st0.5-0.5_hs1.0_bgs2.0_cosine_anneal" "violence" "violence_13class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs12.5_st0.7-0.3_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class_skip #3"

# ============================================================================
# HARASSMENT Evaluation
# ============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}HARASSMENT EVALUATION${NC}"
echo -e "${GREEN}============================================================${NC}"

run_eval "${SAFREE_BASELINE}/harassment" "harassment" "SAFREE Baseline"
run_eval "${SD_BASELINE}/harassment" "harassment" "SD 1.4 Baseline"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "harassment" "harassment_9class #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.7-0.3_hs1.0_bgs2.0_lin" "harassment" "harassment_9class #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.6-0.4_hs1.0_bgs1.0_cos" "harassment" "harassment_9class #3"

run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "harassment" "harassment_9class_skip #1"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.6-0.4_hs1.0_bgs0.0" "harassment" "harassment_9class_skip #2"
run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.4-0.6_hs1.5_bgs2.0_lin" "harassment" "harassment_9class_skip #3"

run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs10.0_st0.3-0.7_hs1.0_bgs0.0" "harassment" "SAFREE_harassment #1"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs10.0_st0.5-0.3_hs1.0_bgs2.0" "harassment" "SAFREE_harassment #2"
run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs5.0_st0.3-0.5_hs1.5_bgs0.0" "harassment" "SAFREE_harassment #3"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o VLM Batch Evaluation Part 1 Complete${NC}"
echo -e "${GREEN}============================================================${NC}"
