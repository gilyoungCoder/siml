#!/bin/bash
# GPT-4o VLM Evaluation for Violence Top 6 Configs
#
# Usage:
#   ./vlm/eval_gpt_violence_top6.sh

set -e
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"


GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/violence_i2p_top6_configs"

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY not set${NC}"
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}GPT-4o Violence Top 6 Evaluation${NC}"
echo -e "${GREEN}============================================================${NC}"

# Function to run evaluation
run_eval() {
    local img_dir=$1
    local desc=$2

    if [ ! -d "$img_dir" ]; then
        echo -e "${YELLOW}Skipping (not found): ${img_dir}${NC}"
        return
    fi

    # Skip if already evaluated
    if [ -f "$img_dir/results_gpt4o_violence.txt" ]; then
        echo -e "${YELLOW}[SKIP] Already evaluated: ${desc}${NC}"
        return
    fi

    echo -e "\n${GREEN}[${desc}]${NC}"
    echo "Directory: ${img_dir}"

    python vlm/gpt_i2p_all.py "$img_dir" "violence"
}

# Top 6 Configs
run_eval "${BASE_DIR}/gs10.0_st0.5-0.3_hs1.0_bgs2.0_cosine_anneal" "Top1: gs10.0_st0.5-0.3_cosine_anneal"
run_eval "${BASE_DIR}/gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease" "Top2: gs10.0_st0.5-0.3_linear_decrease"
run_eval "${BASE_DIR}/gs10.0_st0.3-0.7_hs1.0_bgs1.0_cosine_anneal" "Top3: gs10.0_st0.3-0.7_cosine_anneal"
run_eval "${BASE_DIR}/gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease_skip" "Top4: gs7.5_st0.6-0.4_linear_decrease_skip"
run_eval "${BASE_DIR}/gs12.5_st0.7-0.3_hs1.0_bgs2.0_linear_decrease_skip" "Top5: gs12.5_st0.7-0.3_linear_decrease_skip"
run_eval "${BASE_DIR}/gs7.5_st0.5-0.5_hs1.0_bgs2.0_cosine_anneal_skip" "Top6: gs7.5_st0.5-0.5_cosine_anneal_skip"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}All Done${NC}"
echo -e "${GREEN}============================================================${NC}"
