#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
LOGDIR=$ROOT/logs/i2p_multi_7c_balanced_0501
mkdir -p "$LOGDIR"
CONCEPTS=(sexual violence self-harm shocking illegal_activity harassment hate)
METHODS=(safedenoiser_multi sgf_multi)
GPUS=(0 1 2 3 4 5 6)
next_gpu_idx=0
is_free(){
  local gpu=$1
  local mem util
  read mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ",")
  [ "${mem:-99999}" -lt 1000 ]
}
wait_gpu(){
  while true; do
    for offset in 0 1 2 3 4 5 6; do
      idx=$(( (next_gpu_idx + offset) % 7 ))
      gpu=${GPUS[$idx]}
      if is_free "$gpu"; then
        next_gpu_idx=$(( (idx + 1) % 7 ))
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date)] waiting for free GPU among 0-6; GPU7 reserved" >&2
    sleep 60
  done
}
for method in "${METHODS[@]}"; do
  for concept in "${CONCEPTS[@]}"; do
    gpu=$(wait_gpu)
    log="$LOGDIR/${method}_${concept}_gpu${gpu}.log"
    echo "[$(date)] launch $method $concept on GPU $gpu log=$log"
    nohup "$ROOT/scripts/run_official_i2p_multi_7c_balanced_np.sh" "$method" "$concept" "$gpu" 1 > "$log" 2>&1 &
    sleep 8
  done
done
wait
echo "[$(date)] all balanced 7c jobs finished"
