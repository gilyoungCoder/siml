#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB="$ROOT/joblists/phase2_multi_v5_reeval_60_siml07.tsv"
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GPUS=(1 3 4 5 6 7)
mkdir -p "$ROOT/logs/phase2_v5_reeval_siml07" "$ROOT/pids"
mem_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk 'NR==1{print $1+0}'; }
wait_free(){
  local gpu="$1"; local log="$2"
  while true; do
    m=$(mem_used "$gpu")
    if [[ -n "${m:-}" && "$m" -lt 2000 ]]; then return 0; fi
    echo "[$(date '+%F %T')] gpu=$gpu busy mem=${m:-NA}MiB; waiting" >> "$log"
    sleep 45
  done
}
worker(){
  local idx="$1"; local gpu="$2"; local log="$ROOT/logs/phase2_v5_reeval_siml07/gpu${gpu}.log"
  echo "[$(date '+%F %T')] worker idx=$idx gpu=$gpu start" >> "$log"
  local n=0
  while IFS=$'\t' read -r name source evalc png oldtotal dir; do
    [[ -n "${dir:-}" ]] || continue
    if (( n % ${#GPUS[@]} != idx )); then n=$((n+1)); continue; fi
    n=$((n+1))
    wait_free "$gpu" "$log"
    echo "[$(date '+%F %T')] RUN name=$name source=$source eval=$evalc png=$png oldtotal=$oldtotal dir=$dir" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V5" "$dir" "$evalc" qwen >> "$log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] DONE rc=$rc name=$name eval=$evalc dir=$dir" >> "$log"
    sleep 5
  done < "$JOB"
  echo "[$(date '+%F %T')] worker idx=$idx gpu=$gpu complete" >> "$log"
}
for i in "${!GPUS[@]}"; do
  worker "$i" "${GPUS[$i]}" &
  echo $! > "$ROOT/pids/phase2_v5_reeval_siml07_gpu${GPUS[$i]}.pid"
done
wait
