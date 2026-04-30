#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB=$ROOT/joblists/official_i2p_concept_specific_7concept.tsv
LOG=$ROOT/logs/official_i2p_concept_specific_siml09.log
GPU=0
echo "[$(date '+%F %T')] START concept-specific official queue host=$(hostname) gpu=$GPU" >> "$LOG"
while IFS=$'\t' read -r method concept; do
  [[ -n "${method:-}" ]] || continue
  echo "[$(date '+%F %T')] JOB method=$method concept=$concept" >> "$LOG"
  "$ROOT/scripts/run_official_i2p_concept_specific.sh" "$method" "$concept" "$GPU" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE rc=$rc method=$method concept=$concept" >> "$LOG"
  sleep 3
done < "$JOB"
echo "[$(date '+%F %T')] ALL_DONE concept-specific official queue" >> "$LOG"
