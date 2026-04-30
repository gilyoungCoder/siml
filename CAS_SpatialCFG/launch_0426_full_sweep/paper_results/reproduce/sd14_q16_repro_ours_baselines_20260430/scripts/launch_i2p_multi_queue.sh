#!/usr/bin/env bash
set -uo pipefail
QUEUE=$1; GPU=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
LOG=$ROOT/logs/i2p_multi/$(basename "$QUEUE" .tsv)_$(hostname)_gpu${GPU}.log
mkdir -p "$ROOT/logs/i2p_multi"
echo "[$(date '+%F %T')] START queue=$QUEUE host=$(hostname) gpu=$GPU" >> "$LOG"
while IFS=$'\t' read -r method multi concept; do
 [ -n "${method:-}" ] || continue
 echo "[$(date '+%F %T')] JOB $method $multi $concept" >> "$LOG"
 "$ROOT/scripts/run_official_i2p_multi_np.sh" "$method" "$multi" "$concept" "$GPU" 1 >> "$LOG" 2>&1
 echo "[$(date '+%F %T')] DONE rc=$? $method $multi $concept" >> "$LOG"
 sleep 2
done < "$QUEUE"
echo "[$(date '+%F %T')] ALL_DONE" >> "$LOG"
