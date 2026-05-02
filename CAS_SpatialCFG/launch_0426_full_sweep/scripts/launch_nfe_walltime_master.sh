#!/usr/bin/env bash
# Master dispatch v2: GENERATION ONLY on siml-05 g2..g7 (6 GPUs).
# Timing benchmark dropped (g0 is shared with junhyun).
# Eval is run separately on siml-09 via launch_eval_dispatcher.sh.
#
# Usage: bash launch_nfe_walltime_master.sh

set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/nfe_walltime_v3
mkdir -p $LOGDIR

ORCH=$SCRIPTS/nfe_walltime_orchestrator.py
PYBASE=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
[ -f "$ORCH" ] || { echo "Missing $ORCH"; exit 1; }

GPUS=(2 3 4 5 6 7)   # NEVER 0 (junhyun) or 1
NSLOTS=${#GPUS[@]}

echo "[$(date)] launching $NSLOTS-way generation across siml-05 GPUs ${GPUS[*]} (do_eval=0)"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}; SLOT=$IDX
  LOG=$LOGDIR/master_g${GPU}_s${SLOT}.log
  # do_eval=0 so siml-05 only generates; eval handled on siml-09
  nohup $PYBASE $ORCH $GPU $SLOT $NSLOTS 0 >> $LOG 2>&1 &
  echo "  -> GPU=$GPU slot=$SLOT pid=$! log=$LOG"
done

echo
echo "Generation only. Run eval separately on siml-09:"
echo "  ssh siml-09 'bash $SCRIPTS/launch_eval_dispatcher.sh'"
echo
echo "Monitor:"
echo "  ssh siml-05 'ls $ROOT/outputs/phase_nfe_walltime_v3 | wc -l'"
