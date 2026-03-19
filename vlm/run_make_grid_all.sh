#!/bin/bash
# Generate 4x4 grid images for all best configs
# Save to vlm/grids/ with method names

set -e

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

cd /mnt/home/yhgil99/unlearning/vlm

GREEN='\033[0;32m'
NC='\033[0m'

BASE="/mnt/home/yhgil99/unlearning"
OUT="/mnt/home/yhgil99/unlearning/vlm/grids"

mkdir -p "$OUT"

# NUDITY
echo -e "${GREEN}[Grid] NUDITY${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/sexual" --grid-size 4x4 --title "nudity_SD1.4" --output "${OUT}/nudity_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/sexual" --grid-size 4x4 --title "nudity_SAFREE" --output "${OUT}/nudity_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_4class_nudity/cgs5.0_st0.5-0.3_hs0.5_bgs2.0" --grid-size 4x4 --title "nudity_SAFREE+Ours_4class" --output "${OUT}/nudity_SAFREE+Ours_4class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class/gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal" --grid-size 4x4 --title "nudity_Ours_4class_skip" --output "${OUT}/nudity_Ours_4class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always/gs10.0_thr0.5-0.1_hs1.0_bgs2.0_cosine_anneal" --grid-size 4x4 --title "nudity_Ours_4class_noskip" --output "${OUT}/nudity_Ours_4class_noskip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700/gs10.0_thr0.3-0.3_hs1.5_bgs0.0_cosine_anneal" --grid-size 4x4 --title "nudity_Ours_6class_noskip" --output "${OUT}/nudity_Ours_6class_noskip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip/gs10.0_thr0.3-0.3_hs1.0_bgs0.0_cosine_anneal" --grid-size 4x4 --title "nudity_Ours_6class_skip" --output "${OUT}/nudity_Ours_6class_skip.png" || true

# VIOLENCE
echo -e "${GREEN}[Grid] VIOLENCE${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/violence" --grid-size 4x4 --title "violence_SD1.4" --output "${OUT}/violence_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/violence" --grid-size 4x4 --title "violence_SAFREE" --output "${OUT}/violence_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/violence_step15500/cgs10.0_st0.5-0.3_hs1.0_bgs0.0" --grid-size 4x4 --title "violence_SAFREE+Ours_9class" --output "${OUT}/violence_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500_skip/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" --grid-size 4x4 --title "violence_Ours_9class_skip" --output "${OUT}/violence_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/violence_9class_step15500/gs7.5_st0.6-0.4_hs1.0_bgs2.0_cos" --grid-size 4x4 --title "violence_Ours_9class_noskip" --output "${OUT}/violence_Ours_9class_noskip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/violence_i2p_top6_configs/gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease_skip" --grid-size 4x4 --title "violence_Ours_13class_skip" --output "${OUT}/violence_Ours_13class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/violence_i2p_top6_configs/gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease" --grid-size 4x4 --title "violence_Ours_13class_noskip" --output "${OUT}/violence_Ours_13class_noskip.png" || true

# HARASSMENT
echo -e "${GREEN}[Grid] HARASSMENT${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/harassment" --grid-size 4x4 --title "harassment_SD1.4" --output "${OUT}/harassment_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/harassment" --grid-size 4x4 --title "harassment_SAFREE" --output "${OUT}/harassment_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/harassment_step24300/cgs10.0_st0.5-0.3_hs1.0_bgs2.0" --grid-size 4x4 --title "harassment_SAFREE+Ours_9class" --output "${OUT}/harassment_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300_skip/gs7.5_st0.4-0.6_hs1.5_bgs2.0_lin" --grid-size 4x4 --title "harassment_Ours_9class_skip" --output "${OUT}/harassment_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/harassment_9class_step24300/gs7.5_st0.6-0.4_hs1.0_bgs1.0_cos" --grid-size 4x4 --title "harassment_Ours_9class_noskip" --output "${OUT}/harassment_Ours_9class_noskip.png" || true

# HATE
echo -e "${GREEN}[Grid] HATE${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/hate" --grid-size 4x4 --title "hate_SD1.4" --output "${OUT}/hate_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/hate" --grid-size 4x4 --title "hate_SAFREE" --output "${OUT}/hate_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/hate_step20800/cgs7.5_st0.5-0.5_hs1.5_bgs0.0" --grid-size 4x4 --title "hate_SAFREE+Ours_9class" --output "${OUT}/hate_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800_skip/gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos" --grid-size 4x4 --title "hate_Ours_9class_skip" --output "${OUT}/hate_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/hate_9class_step20800/gs7.5_st0.5-0.3_hs2.0_bgs1.0_cos" --grid-size 4x4 --title "hate_Ours_9class_noskip" --output "${OUT}/hate_Ours_9class_noskip.png" || true

# SHOCKING
echo -e "${GREEN}[Grid] SHOCKING${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/shocking" --grid-size 4x4 --title "shocking_SD1.4" --output "${OUT}/shocking_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/shocking" --grid-size 4x4 --title "shocking_SAFREE" --output "${OUT}/shocking_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/shocking_step23700/cgs7.5_st0.3-0.5_hs1.5_bgs0.0" --grid-size 4x4 --title "shocking_SAFREE+Ours_9class" --output "${OUT}/shocking_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700_skip/gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin" --grid-size 4x4 --title "shocking_Ours_9class_skip" --output "${OUT}/shocking_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/shocking_9class_step23700/gs7.5_st0.7-0.3_hs1.0_bgs1.0_lin" --grid-size 4x4 --title "shocking_Ours_9class_noskip" --output "${OUT}/shocking_Ours_9class_noskip.png" || true

# ILLEGAL
echo -e "${GREEN}[Grid] ILLEGAL${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/illegal_activity" --grid-size 4x4 --title "illegal_SD1.4" --output "${OUT}/illegal_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/illegal activity" --grid-size 4x4 --title "illegal_SAFREE" --output "${OUT}/illegal_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/illegal_step22600/cgs10.0_st0.5-0.5_hs1.5_bgs2.0" --grid-size 4x4 --title "illegal_SAFREE+Ours_9class" --output "${OUT}/illegal_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600_skip/gs10.0_st0.3-0.5_hs1.5_bgs2.0_cos" --grid-size 4x4 --title "illegal_Ours_9class_skip" --output "${OUT}/illegal_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/illegal_9class_step22600/gs7.5_st0.5-0.3_hs1.5_bgs1.0_cos" --grid-size 4x4 --title "illegal_Ours_9class_noskip" --output "${OUT}/illegal_Ours_9class_noskip.png" || true

# SELF_HARM
echo -e "${GREEN}[Grid] SELF_HARM${NC}"
python make_grid.py "${BASE}/SoftDelete+CG/SDbaseline/i2p/selfharm" --grid-size 4x4 --title "selfharm_SD1.4" --output "${OUT}/selfharm_SD1.4.png" || true
python make_grid.py "${BASE}/SAFREE/safree_outputs/i2p/self-harm" --grid-size 4x4 --title "selfharm_SAFREE" --output "${OUT}/selfharm_SAFREE.png" || true
python make_grid.py "${BASE}/SAFREE/results/grid_search_safree_9class/selfharm_step20700/cgs5.0_st0.3-0.7_hs0.5_bgs0.0" --grid-size 4x4 --title "selfharm_SAFREE+Ours_9class" --output "${OUT}/selfharm_SAFREE+Ours_9class.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700_skip/gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin" --grid-size 4x4 --title "selfharm_Ours_9class_skip" --output "${OUT}/selfharm_Ours_9class_skip.png" || true
python make_grid.py "${BASE}/SoftDelete+CG/scg_outputs/grid_search_results/selfharm_9class_step20700/gs7.5_st0.4-0.6_hs1.5_bgs1.0_cos" --grid-size 4x4 --title "selfharm_Ours_9class_noskip" --output "${OUT}/selfharm_Ours_9class_noskip.png" || true

echo -e "${GREEN}ALL GRIDS DONE - saved to ${OUT}${NC}"
ls -la "${OUT}"
