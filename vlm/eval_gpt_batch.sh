#!/bin/bash
# GPT-4o VLM Batch Evaluation Script
# Evaluates top configs from Qwen2-VL analysis + baselines
#
# Usage:
#   ./vlm/eval_gpt_batch.sh [concept]
#   ./vlm/eval_gpt_batch.sh nudity       # Evaluate nudity concept only
#   ./vlm/eval_gpt_batch.sh all          # Evaluate all concepts
#
# Requires: OPENAI_API_KEY environment variable

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"

# Baselines
SAFREE_BASELINE="${BASE_DIR}/SAFREE/safree_outputs/i2p"
SD_BASELINE="${BASE_DIR}/SoftDelete+CG/SDbaseline/i2p"

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set${NC}"
    exit 1
fi

# Concept argument
CONCEPT=${1:-all}

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o VLM Batch Evaluation${NC}"
echo -e "${GREEN}Concept: ${CONCEPT}${NC}"
echo -e "${GREEN}============================================================${NC}"

# Function to run evaluation
run_eval() {
    local img_dir=$1
    local concept=$2
    local desc=$3

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}Skipping (not found): ${img_dir}${NC}"
        return
    fi

    # Skip if already evaluated (GPT-4o results = results_gpt4o_{concept}.txt)
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
eval_nudity() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}NUDITY/SEXUAL EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines
    run_eval "${SAFREE_BASELINE}/sexual" "nudity" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/sexual" "nudity" "SD 1.4 Baseline"

    # nudity_4class_always Top 3 (Best: 100%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.5-0.3_hs1.5_bgs2.0_cosine_anneal" "nudity" "nudity_4class_always #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.5-0.1_hs1.0_bgs2.0_cosine_anneal" "nudity" "nudity_4class_always #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.3-0.5_hs1.5_bgs1.0_cosine_anneal" "nudity" "nudity_4class_always #3"

    # grid_search_nudity_6class_v2_step22700 Top 3 (Best: 100%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.3-0.3_hs0.5_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2 #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.3-0.3_hs1.5_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2 #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.7-0.5_hs0.5_bgs2.0_linear_decrease" "nudity" "nudity_6class_v2 #3"

    # grid_search_nudity_6class_v2_step22700_skip Top 3 (Best: 96%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.3_hs1.0_bgs0.0_cosine_anneal" "nudity" "nudity_6class_v2_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.3_hs1.5_bgs0.0_linear_decrease" "nudity" "nudity_6class_v2_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.5_hs0.5_bgs0.0_linear_decrease" "nudity" "nudity_6class_v2_skip #3"

    # SAFREE 4-class nudity Top 3 (Best: 94%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs5.0_st0.5-0.3_hs0.5_bgs2.0" "nudity" "SAFREE_nudity #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs5.0_st0.5-0.5_hs0.5_bgs2.0" "nudity" "SAFREE_nudity #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_4class_nudity/cgs7.5_st0.7-0.3_hs1.5_bgs0.0" "nudity" "SAFREE_nudity #3"

    # nudity_4class Top 3 (Best: 96%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal" "nudity" "nudity_4class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs10.0_thr0.7-0.3_hs0.5_bgs1.0_cosine_anneal" "nudity" "nudity_4class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs10.0_thr0.7-0.3_hs1.5_bgs2.0_linear_decrease" "nudity" "nudity_4class #3"
}

# ============================================================================
# VIOLENCE Evaluation
# ============================================================================
eval_violence() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}VIOLENCE EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines
    run_eval "${SAFREE_BASELINE}/violence" "violence" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/violence" "violence" "SD 1.4 Baseline"

    # violence_9class_step15500_skip Top 3 (Best: 72%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs12.5_st0.4-0.6_hs1.0_bgs1.0_lin" "violence" "violence_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" "violence" "violence_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs12.5_st0.4-0.6_hs1.0_bgs1.0_cos" "violence" "violence_9class_skip #3"

    # violence_9class_step15500 Top 3 (Best: 60%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "violence" "violence_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" "violence" "violence_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" "violence" "violence_9class #3"

    # SAFREE violence Top 3 (Best: 60%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs5.0_st0.7-0.3_hs1.0_bgs0.0" "violence" "SAFREE_violence #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs5.0_st0.3-0.7_hs0.5_bgs1.0" "violence" "SAFREE_violence #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs10.0_st0.5-0.3_hs1.0_bgs0.0" "violence" "SAFREE_violence #3"

    # violence_13class_step28400 Top 3 (Best: 42%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.3-0.7_hs1.0_bgs1.0_cosine_anneal" "violence" "violence_13class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400/gs10.0_st0.5-0.3_hs1.0_bgs2.0_cosine_anneal" "violence" "violence_13class #3"

    # violence_13class_step28400_skip Top 3 (Best: 42%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs7.5_st0.5-0.5_hs1.0_bgs2.0_cosine_anneal" "violence" "violence_13class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_violence_13class_step28400_skip/gs12.5_st0.7-0.3_hs1.0_bgs2.0_linear_decrease" "violence" "violence_13class_skip #3"
}

# ============================================================================
# HARASSMENT Evaluation
# ============================================================================
eval_harassment() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}HARASSMENT EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines
    run_eval "${SAFREE_BASELINE}/harassment" "harassment" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/harassment" "harassment" "SD 1.4 Baseline"

    # harassment_9class_step24300 Top 3 (Best: 100%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "harassment" "harassment_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.7-0.3_hs1.0_bgs2.0_lin" "harassment" "harassment_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.6-0.4_hs1.0_bgs1.0_cos" "harassment" "harassment_9class #3"

    # harassment_9class_step24300_skip Top 3 (Best: 72.7%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "harassment" "harassment_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.6-0.4_hs1.0_bgs0.0" "harassment" "harassment_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.4-0.6_hs1.5_bgs2.0_lin" "harassment" "harassment_9class_skip #3"

    # SAFREE harassment Top 3 (Best: 54%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs10.0_st0.3-0.7_hs1.0_bgs0.0" "harassment" "SAFREE_harassment #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs10.0_st0.5-0.3_hs1.0_bgs2.0" "harassment" "SAFREE_harassment #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs5.0_st0.3-0.5_hs1.5_bgs0.0" "harassment" "SAFREE_harassment #3"
}

# ============================================================================
# HATE Evaluation
# ============================================================================
eval_hate() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}HATE EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines
    run_eval "${SAFREE_BASELINE}/hate" "hate" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/hate" "hate" "SD 1.4 Baseline"

    # hate_9class_step20800 Top 3 (Best: 32%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs2.0_bgs1.0_cos" "hate" "hate_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs1.0_bgs2.0_lin" "hate" "hate_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.4-0.2_hs1.0_bgs1.0_lin" "hate" "hate_9class #3"

    # hate_9class_step20800_skip Top 3 (Best: 30%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs12.5_st0.5-0.5_hs1.5_bgs2.0_cos" "hate" "hate_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos" "hate" "hate_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs7.5_st0.2-0.4_hs2.0_bgs2.0_cos" "hate" "hate_9class_skip #3"

    # SAFREE hate Top 3 (Best: 10%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs7.5_st0.5-0.5_hs1.5_bgs0.0" "hate" "SAFREE_hate #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs5.0_st0.7-0.3_hs1.5_bgs1.0" "hate" "SAFREE_hate #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs5.0_st0.5-0.5_hs1.5_bgs1.0" "hate" "SAFREE_hate #3"
}

# ============================================================================
# SHOCKING Evaluation
# ============================================================================
eval_shocking() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}SHOCKING EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines
    run_eval "${SAFREE_BASELINE}/shocking" "shocking" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/shocking" "shocking" "SD 1.4 Baseline"

    # shocking_9class_step23700_skip Top 3 (Best: 90.9%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs7.5_st0.6-0.4_hs1.0_bgs1.0" "shocking" "shocking_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" "shocking" "shocking_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" "shocking" "shocking_9class_skip #3"

    # shocking_9class_step23700 Top 3 (Best: 58%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_cos" "shocking" "shocking_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_lin" "shocking" "shocking_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.4-0.6_hs1.0_bgs1.0_cos" "shocking" "shocking_9class #3"

    # SAFREE shocking Top 3 (Best: 70%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.7_hs1.0_bgs2.0" "shocking" "SAFREE_shocking #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.5_hs1.5_bgs0.0" "shocking" "SAFREE_shocking #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs5.0_st0.7-0.3_hs1.0_bgs2.0" "shocking" "SAFREE_shocking #3"
}

# ============================================================================
# ILLEGAL Evaluation
# ============================================================================
eval_illegal() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}ILLEGAL EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines (Note: different folder names)
    run_eval "${SAFREE_BASELINE}/illegal activity" "illegal" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/illegal_activity" "illegal" "SD 1.4 Baseline"

    # illegal_9class_step22600_skip Top 3 (Best: 30%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.7_hs1.5_bgs2.0_cos" "illegal" "illegal_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs12.5_st0.2-0.4_hs1.0_bgs2.0_cos" "illegal" "illegal_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.5_hs1.5_bgs2.0_cos" "illegal" "illegal_9class_skip #3"

    # illegal_9class_step22600 Top 3 (Best: 26%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.0_bgs2.0_lin" "illegal" "illegal_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" "illegal" "illegal_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs10.0_st0.5-0.5_hs1.5_bgs1.0_lin" "illegal" "illegal_9class #3"

    # SAFREE illegal Top 3 (Best: 22%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs5.0_st0.5-0.3_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs5.0_st0.5-0.5_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs10.0_st0.5-0.5_hs1.5_bgs2.0" "illegal" "SAFREE_illegal #3"
}

# ============================================================================
# SELF-HARM Evaluation
# ============================================================================
eval_selfharm() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}SELF-HARM EVALUATION${NC}"
    echo -e "${GREEN}============================================================${NC}"

    # Baselines (Note: different folder names)
    run_eval "${SAFREE_BASELINE}/self-harm" "self_harm" "SAFREE Baseline"
    run_eval "${SD_BASELINE}/selfharm" "self_harm" "SD 1.4 Baseline"

    # selfharm_9class_step20700 Top 3 (Best: 8%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" "self_harm" "selfharm_9class #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.0_bgs1.0_lin" "self_harm" "selfharm_9class #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.5_bgs1.0_cos" "self_harm" "selfharm_9class #3"

    # selfharm_9class_step20700_skip Top 3 (Best: 6%)
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.5-0.3_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #1"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.5-0.5_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #2"
    run_eval "${BASE_DIR}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" "self_harm" "selfharm_9class_skip #3"

    # SAFREE selfharm Top 3 (Best: 4%)
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs0.5_bgs0.0" "self_harm" "SAFREE_selfharm #1"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs1.0_bgs0.0" "self_harm" "SAFREE_selfharm #2"
    run_eval "${BASE_DIR}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs1.0_bgs1.0" "self_harm" "SAFREE_selfharm #3"
}

# Main execution
case $CONCEPT in
    nudity|sexual)
        eval_nudity
        ;;
    violence)
        eval_violence
        ;;
    harassment)
        eval_harassment
        ;;
    hate)
        eval_hate
        ;;
    shocking)
        eval_shocking
        ;;
    illegal)
        eval_illegal
        ;;
    self_harm|selfharm)
        eval_selfharm
        ;;
    all)
        eval_nudity
        eval_violence
        eval_harassment
        eval_hate
        eval_shocking
        eval_illegal
        eval_selfharm
        ;;
    *)
        echo -e "${RED}Unknown concept: ${CONCEPT}${NC}"
        echo "Available: nudity, violence, harassment, hate, shocking, illegal, self_harm, all"
        exit 1
        ;;
esac

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o VLM Batch Evaluation Complete${NC}"
echo -e "${GREEN}============================================================${NC}"
