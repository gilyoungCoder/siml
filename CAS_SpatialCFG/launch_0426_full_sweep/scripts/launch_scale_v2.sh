#!/usr/bin/env bash
# Scale-robustness v2 master: 4 SLD variants × sexual × gs sweep on g1..g5.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/scale_robustness_v2
mkdir -p $LOGDIR

ORCH=$SCRIPTS/scale_v2_orchestrator.py
PYBASE=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
[ -f "$ORCH" ] || { echo "Missing $ORCH"; exit 1; }

GPUS=(1 2 3 4 5)   # NEVER g0 (junhyun); g6/g7 (other users)
NSLOTS=${#GPUS[@]}

echo "[$(date)] launching scale-v2 $NSLOTS-way on siml-05 GPUs ${GPUS[*]}"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}; SLOT=$IDX
  LOG=$LOGDIR/master_g${GPU}_s${SLOT}.log
  nohup $PYBASE $ORCH $GPU $SLOT $NSLOTS >> $LOG 2>&1 &
  echo "  -> GPU=$GPU slot=$SLOT pid=$!"
done
