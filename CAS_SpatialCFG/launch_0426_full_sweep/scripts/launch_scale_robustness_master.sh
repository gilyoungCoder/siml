#!/usr/bin/env bash
# Scale-robustness master dispatcher: 6 parallel slots on siml-05 g2..g7 (NEVER g0/g1).
# Stacks on top of any other yhgil99 jobs already on the same GPUs.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/scale_robustness
mkdir -p $LOGDIR

ORCH=$SCRIPTS/scale_robustness_orchestrator.py
PYBASE=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

[ -f "$ORCH" ] || { echo "Missing $ORCH"; exit 1; }

GPUS=(2 3 4 5 6 7)   # NEVER 0 or 1
NSLOTS=${#GPUS[@]}

echo "[$(date)] launching scale-robustness $NSLOTS-way on siml-05 GPUs ${GPUS[*]}"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}; SLOT=$IDX
  LOG=$LOGDIR/master_g${GPU}_s${SLOT}.log
  nohup $PYBASE $ORCH $GPU $SLOT $NSLOTS >> $LOG 2>&1 &
  echo "  -> GPU=$GPU slot=$SLOT pid=$! log=$LOG"
done

echo
echo "Monitor:"
echo "  ls $ROOT/outputs/phase_scale_robustness | wc -l"
echo "  find $ROOT/outputs/phase_scale_robustness -name 'results_qwen3_vl_*_v5.txt' | wc -l"
