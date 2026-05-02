#!/usr/bin/env bash
# Scale-robustness master (gen only, no eval): siml-05 g2..g7.
# Eval is dispatched separately on siml-09.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/scale_robustness
mkdir -p $LOGDIR

ORCH=$SCRIPTS/scale_robustness_orchestrator.py
PYBASE=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
[ -f "$ORCH" ] || { echo "Missing $ORCH"; exit 1; }

GPUS=(2 3 4 5 6 7)
NSLOTS=${#GPUS[@]}

echo "[$(date)] launching scale-robustness $NSLOTS-way on siml-05 GPUs ${GPUS[*]} (gen only)"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}; SLOT=$IDX
  LOG=$LOGDIR/master_g${GPU}_s${SLOT}.log
  nohup $PYBASE $ORCH $GPU $SLOT $NSLOTS >> $LOG 2>&1 &
  echo "  -> GPU=$GPU slot=$SLOT pid=$! log=$LOG"
done

echo
echo "Eval is dispatched separately on siml-09:"
echo "  ssh siml-09 'bash $SCRIPTS/launch_eval_dispatcher.sh'"
