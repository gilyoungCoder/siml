#!/usr/bin/env bash
set -uo pipefail
QUEUE=$1
GPU=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
LOG=$ROOT/logs/i2pnp/$(basename "$QUEUE" .tsv)_$(hostname)_gpu${GPU}.log
mkdir -p "$ROOT/logs/i2pnp"
echo "[$(date '+%F %T')] START queue=$QUEUE host=$(hostname) gpu=$GPU" >> "$LOG"
while IFS=$'\t' read -r method concept; do
  [[ -n "${method:-}" ]] || continue
  echo "[$(date '+%F %T')] JOB method=$method concept=$concept" >> "$LOG"
  "$ROOT/scripts/run_official_i2p_concept_np.sh" "$method" "$concept" "$GPU" 1 >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE rc=$rc method=$method concept=$concept" >> "$LOG"
  sleep 2
done < "$QUEUE"
echo "[$(date '+%F %T')] ALL_DONE queue=$QUEUE" >> "$LOG"
