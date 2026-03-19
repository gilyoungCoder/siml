#!/bin/bash
# Evaluate remaining folders with Q16 on GPU 6

set -e

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

export CUDA_VISIBLE_DEVICES=6

GREEN='\033[0;32m'
NC='\033[0m'

BASE="/mnt/home/yhgil99/unlearning"
cd /mnt/home/yhgil99/unlearning/vlm

# HARASSMENT - Ours configs
echo -e "${GREEN}[Q16] Harassment Ours configs${NC}"
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.7-0.3_hs1.0_bgs2.0_lin" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.4-0.6_hs1.5_bgs2.0_lin" --threshold 0.5

# HATE - All configs
echo -e "${GREEN}[Q16] Hate configs${NC}"
python eval_q16.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/hate" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/safree_outputs/i2p/hate" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs7.5_st0.5-0.5_hs1.5_bgs0.0" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs2.0_bgs1.0_cos" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos" --threshold 0.5

# SHOCKING - All configs
echo -e "${GREEN}[Q16] Shocking configs${NC}"
python eval_q16.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/shocking" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/safree_outputs/i2p/shocking" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.5_hs1.5_bgs0.0" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_lin" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" --threshold 0.5

# ILLEGAL - All configs
echo -e "${GREEN}[Q16] Illegal configs${NC}"
python eval_q16.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/illegal_activity" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/safree_outputs/i2p/illegal activity" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs10.0_st0.5-0.5_hs1.5_bgs2.0" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.5_hs1.5_bgs2.0_cos" --threshold 0.5

# SELF_HARM - All configs
echo -e "${GREEN}[Q16] Self-harm configs${NC}"
python eval_q16.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/selfharm" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/safree_outputs/i2p/self-harm" --threshold 0.5
python eval_q16.py "${BASE}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs0.5_bgs0.0" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.5_bgs1.0_cos" --threshold 0.5
python eval_q16.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" --threshold 0.5

echo -e "${GREEN}ALL DONE${NC}"
