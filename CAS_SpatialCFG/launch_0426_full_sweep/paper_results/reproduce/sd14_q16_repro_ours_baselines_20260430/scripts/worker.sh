#!/bin/bash
set -uo pipefail
GPU=$1
JOBLIST=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
LOGPFX="[worker gpu=$GPU joblist=$(basename "$JOBLIST")]"
echo "$LOGPFX START $(date) host=$(hostname)"
while IFS=$'\t' read -r KIND A B; do
  [[ -z "${KIND:-}" || "$KIND" =~ ^# ]] && continue
  echo "$LOGPFX JOB kind=$KIND a=$A b=$B start=$(date)"
  case "$KIND" in
    ours)
      "$ROOT/scripts/run_ours_from_config.py" --config "$A" --gpu "$GPU" ;;
    txt)
      "$ROOT/scripts/run_txt_method.sh" "$A" "$B" "$GPU" ;;
    official)
      "$ROOT/scripts/run_official_i2p.sh" "$A" "$B" "$GPU" ;;
    *) echo "$LOGPFX unknown kind=$KIND"; exit 2 ;;
  esac
  rc=$?
  echo "$LOGPFX JOB_DONE kind=$KIND a=$A b=$B rc=$rc end=$(date)"
  if [ $rc -ne 0 ]; then echo "$LOGPFX CONTINUE after failure rc=$rc"; fi
done < "$JOBLIST"
echo "$LOGPFX DONE $(date)"
