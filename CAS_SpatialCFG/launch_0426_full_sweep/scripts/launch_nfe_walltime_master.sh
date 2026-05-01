#!/usr/bin/env bash
# Master dispatch: kicks off generation+eval on siml-05 g2..g7 (6 GPUs) in parallel
# AND the timing benchmark on g0. Run from siml-05 directly.
#
# Usage: bash launch_nfe_walltime_master.sh

set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/nfe_walltime_v3
mkdir -p $LOGDIR

ORCH=$SCRIPTS/nfe_walltime_orchestrator.py
TIMING=$SCRIPTS/nfe_walltime_timing.sh
PYBASE=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

# Sanity
[ -f "$ORCH" ] || { echo "Missing $ORCH"; exit 1; }
[ -f "$TIMING" ] || { echo "Missing $TIMING"; exit 1; }

GPUS=(2 3 4 5 6 7)   # 6 generation GPUs (siml-05 g0 reserved for timing, g1 user)
NSLOTS=${#GPUS[@]}

echo "[$(date)] launching $NSLOTS-way generation across siml-05 GPUs ${GPUS[*]}"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}
  SLOT=$IDX
  LOG=$LOGDIR/master_g${GPU}_s${SLOT}.log
  nohup $PYBASE $ORCH $GPU $SLOT $NSLOTS 1 >> $LOG 2>&1 &
  echo "  -> GPU=$GPU slot=$SLOT pid=$! log=$LOG"
done

echo "[$(date)] launching timing benchmark on g0 (foreground)"
TIMING_LOG=$LOGDIR/timing_master.log
nohup bash $TIMING 0 >> $TIMING_LOG 2>&1 &
echo "  -> timing pid=$! log=$TIMING_LOG"

echo
echo "[$(date)] all jobs launched. Tail logs with:"
echo "  tail -F $LOGDIR/*.log"
echo
echo "Monitor progress:"
echo "  ls $ROOT/outputs/phase_nfe_walltime_v3 | wc -l   # cells started"
echo "  find $ROOT/outputs/phase_nfe_walltime_v3 -name 'results_qwen3_vl_*_v5.txt' | wc -l   # cells eval'd"
