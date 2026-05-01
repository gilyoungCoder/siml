#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB=$ROOT/joblists/sd3_completed_missing_eval_0501.tsv
LOGDIR=$ROOT/logs/eval_sd3_completed_missing_v5_siml07_0501
mkdir -p "$LOGDIR" "$ROOT/pids"
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLM=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GPUS=(1 2 4 5 6)
mem_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk "NR==1{print \\$1+0}"; }
wait_free(){ local g=$1; while true; do m=$(mem_used "$g"); [ "$m" -lt 3000 ] && return 0; echo "$(date +%F_%T) gpu=$g busy mem=$m"; sleep 60; done; }
worker(){ local idx=$1 gpu=$2 log=$LOGDIR/gpu${gpu}.log; local n=0; echo "START idx=$idx gpu=$gpu $(date)" >> "$log"; while IFS=$t read -r method sub name concept dir; do n=$((n+1)); [ $(((n-1)%${#GPUS[@]})) -eq "$idx" ] || continue; res="$dir/results_qwen3_vl_${concept}_v5.txt"; cat="$dir/categories_qwen3_vl_${concept}_v5.json"; if [ -s "$res" ] || [ -s "$cat" ]; then echo "SKIP exists $method $sub $name" >> "$log"; continue; fi; wait_free "$gpu" >> "$log" 2>&1; echo "RUN $method $sub $name concept=$concept dir=$dir $(date)" >> "$log"; CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$VLM" "$dir" "$concept" qwen >> "$log" 2>&1; echo "DONE rc=$? $method $sub $name $(date)" >> "$log"; sleep 5; done < "$JOB"; echo "COMPLETE idx=$idx gpu=$gpu $(date)" >> "$log"; }
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & echo $! > "$ROOT/pids/eval_sd3_completed_missing_v5_gpu${GPUS[$i]}.pid"; done
wait
