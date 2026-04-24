#!/bin/bash
# Launch NudeNet+Q16 on 8 best Ours nudity cells across siml-01 g0-g7.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_nudenet_q16/eval_nudenet_q16_worker.sh
LOGDIR=$REPO/logs/launch_0424_nudenet_q16
mkdir -p $LOGDIR
OUT=$REPO/CAS_SpatialCFG/outputs

# (gpu, label, cell_subdir)
JOBS=(
  "0 ud_anchor   launch_0420_nudity/ours_sd14_v2pack/unlearndiff/anchor_ss1.2_thr0.1_imgthr0.3_both"
  "1 ud_hybrid   launch_0420_nudity/ours_sd14_v1pack/unlearndiff/hybrid_ss10_thr0.1_imgthr0.3_both"
  "2 rab_anchor  launch_0424_rab_anchor_v2pack/anchor_ss1.2_thr0.1_imgthr0.3_both"
  "3 rab_hybrid  launch_0420_nudity/ours_sd14_v2pack/rab/hybrid_ss20_thr0.1_imgthr0.4_both"
  "4 mma_anchor  launch_0420_nudity/ours_sd14_v2pack/mma/anchor_ss1.2_thr0.1_imgthr0.3_both"
  "5 mma_hybrid  launch_0420_nudity/ours_sd14_v1pack/mma/hybrid_ss20_thr0.1_imgthr0.3_both"
  "6 p4dn_anchor launch_0420_nudity/ours_sd14_v2pack/p4dn/anchor_ss1.2_thr0.1_imgthr0.3_both"
  "7 p4dn_hybrid launch_0420_nudity/ours_sd14_v1pack/p4dn/hybrid_ss20_thr0.1_imgthr0.3_both"
)

for spec in "${JOBS[@]}"; do
  read GPU LABEL SUB <<< "$spec"
  nohup bash $SCRIPT $GPU $OUT/$SUB $LABEL > $LOGDIR/launch_g${GPU}.out 2>&1 &
  echo "Launched g$GPU $LABEL"
done
wait
echo "[$(date)] All 8 NudeNet+Q16 evals done"
