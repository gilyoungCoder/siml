#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB="$ROOT/joblists/eval_v5_safree_generated_siml09.tsv"
LOG="$ROOT/logs/eval_v5_safree_generated_siml09.log"
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GPU=0
echo "[$(date '+%F %T')] START safree generated v5 eval on $(hostname) gpu=$GPU" >> "$LOG"
while IFS=$'\t' read -r group item concept expected dir; do
  [[ -n "$dir" ]] || continue
  cat_json="$dir/categories_qwen3_vl_${concept}_v5.json"
  res_txt="$dir/results_qwen3_vl_${concept}_v5.txt"
  pngs=$(find "$dir" -maxdepth 1 -name '*.png' | wc -l | awk '{print $1}')
  echo "[$(date '+%F %T')] CHECK group=$group item=$item concept=$concept pngs=$pngs expected=$expected dir=$dir" >> "$LOG"
  if [[ -s "$cat_json" || -s "$res_txt" ]]; then
    echo "[$(date '+%F %T')] SKIP existing result $dir" >> "$LOG"
    continue
  fi
  if [[ "$pngs" -lt 1 ]]; then
    echo "[$(date '+%F %T')] SKIP no png $dir" >> "$LOG"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$V5" "$dir" "$concept" qwen >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE rc=$rc group=$group item=$item concept=$concept dir=$dir" >> "$LOG"
  sleep 5
done < "$JOB"
echo "[$(date '+%F %T')] ALL_DONE" >> "$LOG"
