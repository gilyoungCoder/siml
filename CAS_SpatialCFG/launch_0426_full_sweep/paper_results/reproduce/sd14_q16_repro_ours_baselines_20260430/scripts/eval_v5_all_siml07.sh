#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOBLIST="$ROOT/joblists/eval_v5_all_siml07.tsv"
LOGDIR="$ROOT/logs/eval_v5_all_siml07"
mkdir -p "$LOGDIR" "$ROOT/pids"
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GPUS=(1 3 4 5 6 7)
mem_used() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk 'NR==1{print $1+0}'
}
wait_free() {
  local gpu="$1"
  while true; do
    local m; m=$(mem_used "$gpu")
    if [[ -n "$m" && "$m" -lt 2000 ]]; then return 0; fi
    echo "[$(date '+%F %T')] gpu=$gpu busy mem=${m:-NA}MiB; waiting" 
    sleep 60
  done
}
worker() {
  local idx="$1" gpu="$2"
  local log="$LOGDIR/gpu${gpu}.log"
  echo "[$(date '+%F %T')] worker idx=$idx gpu=$gpu start" >> "$log"
  local n=0
  while IFS=$'\t' read -r group concept pngs expected dir; do
    n=$((n+1))
    # static split by line number across workers
    if (( (n-1) % ${#GPUS[@]} != idx )); then continue; fi
    local cat="$dir/categories_qwen3_vl_${concept}_v5.json"
    local txt="$dir/results_qwen3_vl_${concept}_v5.txt"
    if [[ -s "$cat" || -s "$txt" ]]; then
      echo "[$(date '+%F %T')] SKIP exists group=$group concept=$concept dir=$dir" >> "$log"
      continue
    fi
    if [[ ! -d "$dir" ]]; then
      echo "[$(date '+%F %T')] SKIP missing dir=$dir" >> "$log"
      continue
    fi
    wait_free "$gpu" >> "$log" 2>&1
    echo "[$(date '+%F %T')] RUN group=$group concept=$concept pngs=$pngs expected=$expected dir=$dir" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V5" "$dir" "$concept" qwen >> "$log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] DONE rc=$rc group=$group concept=$concept dir=$dir" >> "$log"
    # brief cooldown to release memory cleanly
    sleep 5
  done < "$JOBLIST"
  echo "[$(date '+%F %T')] worker idx=$idx gpu=$gpu complete" >> "$log"
}
for i in "${!GPUS[@]}"; do
  worker "$i" "${GPUS[$i]}" &
  echo $! > "$ROOT/pids/eval_v5_all_siml07_gpu${GPUS[$i]}.pid"
done
wait
