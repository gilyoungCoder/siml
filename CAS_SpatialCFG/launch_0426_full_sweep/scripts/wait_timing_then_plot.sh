#!/usr/bin/env bash
# Watchdog: polls timing CSV until row count >= TARGET, then runs both plots.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOG=$ROOT/logs/wait_timing_then_plot_$(date +%m%d_%H%M).log
TIMING=$ROOT/paper_results/figures/nfe_walltime_timing.csv
TARGET=57   # header + 7 method × 8 NFE
TIMEOUT_SEC=2700  # 45 min cap

log_msg () { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
start_t=$(date +%s)
log_msg "watching $TIMING for >=$TARGET lines"

while :; do
  cur=$(wc -l < "$TIMING" 2>/dev/null || echo 0)
  elapsed=$(( $(date +%s) - start_t ))
  if [ "$cur" -ge "$TARGET" ]; then
    log_msg "DONE: $cur lines, elapsed=${elapsed}s → plotting"
    break
  fi
  if [ "$elapsed" -ge "$TIMEOUT_SEC" ]; then
    log_msg "TIMEOUT after ${elapsed}s; $cur/$TARGET lines → plotting partial"
    break
  fi
  log_msg "$cur/$TARGET lines (waited ${elapsed}s)"
  sleep 60
done

log_msg "running NFE plot ..."
$PY $ROOT/scripts/nfe_walltime_pareto_polished.py 2>&1 | tee -a $LOG
log_msg "running scale plot ..."
$PY $ROOT/scripts/scale_robustness_plot.py 2>&1 | tee -a $LOG
log_msg "plots saved → $ROOT/paper_results/figures/"
log_msg "exiting"
