#!/bin/bash
# Launch NudeNet+Q16 @0.7 on 8 baseline+SAFREE nudity cells (siml-01 g0-g7).
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_nudenet_q16/eval_nudenet_q16_worker.sh
LOGDIR=$REPO/logs/launch_0424_nudenet_q16
mkdir -p $LOGDIR
OUT=$REPO/CAS_SpatialCFG/outputs

JOBS=(
  "0 base_ud   launch_0420_nudity/baseline_sd14/unlearndiff"
  "1 base_rab  launch_0420_nudity/baseline_sd14/rab"
  "2 base_mma  launch_0420_nudity/baseline_sd14/mma"
  "3 base_p4dn launch_0420_nudity/baseline_sd14/p4dn"
  "4 sa_ud     launch_0420_nudity/safree_sd14/unlearndiff"
  "5 sa_rab    launch_0420_nudity/safree_sd14/rab"
  "6 sa_mma    launch_0420_nudity/safree_sd14/mma"
  "7 sa_p4dn   launch_0420_nudity/safree_sd14/p4dn"
)
for spec in "${JOBS[@]}"; do
  read GPU LABEL SUB <<< "$spec"
  nohup bash $SCRIPT $GPU $OUT/$SUB $LABEL > $LOGDIR/launch_baseline_g${GPU}.out 2>&1 &
  echo "Launched g$GPU $LABEL"
done
wait
echo "[$(date)] All 8 baseline+SAFREE NudeNet+Q16 evals done"
