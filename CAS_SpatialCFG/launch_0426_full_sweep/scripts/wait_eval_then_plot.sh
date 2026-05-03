#!/usr/bin/env bash
# Watchdog: polls eval completion every 60s, then runs both plot scripts.
# Targets: NFE 504, Scale 28. Time-out 30 min.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOG=$ROOT/logs/wait_eval_then_plot_$(date +%m%d_%H%M).log
NFE_TARGET=504
SCALE_TARGET=28
TIMEOUT_SEC=1800

log_msg () { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

start_t=$(date +%s)
log_msg "watching eval completion (NFE>=$NFE_TARGET, Scale>=$SCALE_TARGET)"

while :; do
  nfe_done=$(find $ROOT/outputs/phase_nfe_walltime_v3 -name 'results_qwen3_vl_*_v5.txt' -size +50c 2>/dev/null | wc -l)
  scale_done=$(find $ROOT/outputs/phase_scale_robustness -name 'results_qwen3_vl_*_v5.txt' -size +50c 2>/dev/null | wc -l)
  elapsed=$(( $(date +%s) - start_t ))
  if [ "$nfe_done" -ge "$NFE_TARGET" ] && [ "$scale_done" -ge "$SCALE_TARGET" ]; then
    log_msg "DONE: nfe=$nfe_done scale=$scale_done elapsed=${elapsed}s → plotting"
    break
  fi
  if [ "$elapsed" -ge "$TIMEOUT_SEC" ]; then
    log_msg "TIMEOUT after ${elapsed}s; nfe=$nfe_done/$NFE_TARGET scale=$scale_done/$SCALE_TARGET → plotting partial"
    break
  fi
  log_msg "pending: nfe=$nfe_done/$NFE_TARGET scale=$scale_done/$SCALE_TARGET (waited ${elapsed}s)"
  sleep 60
done

log_msg "running NFE plot ..."
$PY $ROOT/scripts/nfe_walltime_pareto_polished.py 2>&1 | tee -a $LOG
log_msg "running scale plot ..."
$PY $ROOT/scripts/scale_robustness_plot.py 2>&1 | tee -a $LOG
log_msg "all plots saved → $ROOT/paper_results/figures/"
ls -la $ROOT/paper_results/figures/nfe_walltime_pareto_*.png $ROOT/paper_results/figures/scale_robustness*.png 2>&1 | tail -20 | tee -a $LOG
log_msg "watchdog exiting"
