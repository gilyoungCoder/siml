#!/usr/bin/env bash
# Watchdog: polls siml-05 g2..g7 until ALL six are idle (no significant memory
# usage by any process), then triggers the SLD-only timing benchmark on g2 alone.
# Sleeps 5 min between polls. Designed to be run as nohup background on siml-05.
#
# Usage: nohup bash wait_then_sld_timing.sh > wait_then_sld_timing.log 2>&1 &

set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
TIMING=$ROOT/scripts/sld_timing_benchmark.sh
WATCH_GPUS=(2 3 4 5 6 7)
TIMING_GPU=2          # the single GPU used for actual timing measurement
IDLE_THRESHOLD_MIB=300  # below this we consider the GPU "idle"
POLL_SEC=300            # 5 min between polls
LOG=$ROOT/logs/wait_then_sld_timing_$(date +%m%d_%H%M).log
mkdir -p $ROOT/logs

log_msg () { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log_msg "watchdog started; polling g${WATCH_GPUS[*]} every ${POLL_SEC}s; idle threshold ${IDLE_THRESHOLD_MIB} MiB"

iterations=0
while :; do
  iterations=$((iterations+1))
  # Get memory.used per GPU (single line per GPU)
  declare -A USED
  while IFS=',' read -r idx mib; do
    idx=$(echo "$idx" | tr -d ' ')
    mib=$(echo "$mib" | tr -d ' MiB')
    USED[$idx]=$mib
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  all_idle=1
  status_str=""
  for g in "${WATCH_GPUS[@]}"; do
    used="${USED[$g]:-?}"
    status_str+="g${g}=${used}MiB "
    if [ "$used" -ge $IDLE_THRESHOLD_MIB ] 2>/dev/null; then
      all_idle=0
    fi
  done

  if [ "$all_idle" -eq 1 ]; then
    log_msg "ALL IDLE detected: $status_str → starting timing on g$TIMING_GPU"
    bash $TIMING $TIMING_GPU 2>&1 | tee -a $LOG
    log_msg "timing finished; watchdog exiting"
    exit 0
  fi

  if [ $((iterations % 6)) -eq 1 ]; then
    log_msg "still busy: $status_str"
  fi
  sleep $POLL_SEC
done
