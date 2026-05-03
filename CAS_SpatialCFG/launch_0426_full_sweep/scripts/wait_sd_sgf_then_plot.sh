#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOG=$ROOT/logs/wait_sd_sgf_then_plot_$(date +%m%d_%H%M).log
TIMEOUT=2700
log_msg () { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
start_t=$(date +%s)
log_msg "watching SD + SGF clean CSVs (8 NFE rows each)"
while :; do
  done_count=0
  for M in safedenoiser sgf; do
    F=$ROOT/paper_results/figures/nfe_walltime_timing_clean_${M}.csv
    if [ -f "$F" ] && [ "$(tail -n +2 "$F" | wc -l)" -ge 8 ]; then
      done_count=$((done_count+1))
    fi
  done
  elapsed=$(( $(date +%s) - start_t ))
  if [ "$done_count" -ge 2 ]; then log_msg "ALL DONE (${elapsed}s)"; break; fi
  if [ "$elapsed" -ge "$TIMEOUT" ]; then log_msg "TIMEOUT $done_count/2"; break; fi
  log_msg "$done_count/2 (${elapsed}s)"
  sleep 60
done
log_msg "merging + plotting"
bash $ROOT/scripts/merge_timing_clean.sh 2>&1 | tee -a $LOG
log_msg "DONE"
