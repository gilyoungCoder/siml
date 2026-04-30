#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB=$ROOT/joblists/eval_v5_cs_official.tsv
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
LOG=$ROOT/logs/eval_v5_cs_official_siml09.log
GPU=${1:-0}
echo "[$(date)] START cs official eval gpu=$GPU" >> "$LOG"
while IFS=$'\t' read -r method concept eval_concept dir; do
  [ -n "${dir:-}" ] || continue
  res="$dir/results_qwen3_vl_${eval_concept}_v5.txt"
  cat="$dir/categories_qwen3_vl_${eval_concept}_v5.json"
  actual=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ "$actual" -lt 60 ]; then echo "[$(date)] SKIP incomplete $method/$concept count=$actual dir=$dir" >> "$LOG"; continue; fi
  if [ -s "$res" ] || [ -s "$cat" ]; then echo "[$(date)] SKIP existing $method/$concept" >> "$LOG"; continue; fi
  echo "[$(date)] RUN $method/$concept eval=$eval_concept count=$actual" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$V5" "$dir" "$eval_concept" qwen >> "$LOG" 2>&1
  echo "[$(date)] DONE rc=$? $method/$concept" >> "$LOG"
  sleep 3
done < "$JOB"
echo "[$(date)] ALL_DONE" >> "$LOG"
