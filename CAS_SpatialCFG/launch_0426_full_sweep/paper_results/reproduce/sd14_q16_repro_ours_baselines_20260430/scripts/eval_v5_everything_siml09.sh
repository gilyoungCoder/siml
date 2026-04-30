#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB="$ROOT/joblists/eval_v5_everything_siml09.tsv"
LOG="$ROOT/logs/eval_v5_everything_siml09.log"
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GPU=0
mem_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | awk 'NR==1{print $1+0}'; }
wait_free(){ while true; do m=$(mem_used); if [[ -n "${m:-}" && "$m" -lt 2000 ]]; then return 0; fi; echo "[$(date '+%F %T')] gpu=$GPU busy mem=${m:-NA}MiB; waiting" >> "$LOG"; sleep 45; done; }
echo "[$(date '+%F %T')] START everything v5 eval FIXED on $(hostname) gpu=$GPU job=$JOB" >> "$LOG"
while IFS=$'\t' read -r group source_concept eval_concept pngs expected dir; do
  [[ -n "${dir:-}" ]] || { echo "[$(date '+%F %T')] BAD_LINE group=$group source=$source_concept eval=$eval_concept pngs=$pngs expected=$expected dir=${dir:-}" >> "$LOG"; continue; }
  res="$dir/results_qwen3_vl_${eval_concept}_v5.txt"; cat="$dir/categories_qwen3_vl_${eval_concept}_v5.json"
  if [[ -s "$res" || -s "$cat" ]]; then echo "[$(date '+%F %T')] SKIP existing group=$group eval=$eval_concept dir=$dir" >> "$LOG"; continue; fi
  [[ -d "$dir" ]] || { echo "[$(date '+%F %T')] SKIP missing dir=$dir" >> "$LOG"; continue; }
  actual=$(find "$dir" -maxdepth 1 -name '*.png' | wc -l | awk '{print $1}')
  [[ "$actual" -gt 0 ]] || { echo "[$(date '+%F %T')] SKIP no_png dir=$dir" >> "$LOG"; continue; }
  wait_free
  echo "[$(date '+%F %T')] RUN group=$group source=$source_concept eval=$eval_concept pngs=$actual expected=$expected dir=$dir" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$V5" "$dir" "$eval_concept" qwen >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE rc=$rc group=$group eval=$eval_concept dir=$dir" >> "$LOG"
  sleep 5
done < "$JOB"
echo "[$(date '+%F %T')] ALL_DONE everything v5 eval" >> "$LOG"
