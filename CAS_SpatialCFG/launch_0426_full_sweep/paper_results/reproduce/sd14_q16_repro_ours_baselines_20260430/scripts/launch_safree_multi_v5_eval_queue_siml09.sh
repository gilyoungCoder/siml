#!/usr/bin/env bash
set -uo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_safree_multi_q16top60
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/safree_multi_v5_eval_0501
mkdir -p "$LOGDIR"
wait_gpu0(){
  while true; do
    read mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ',')
    if [ "${mem:-99999}" -lt 1000 ]; then return 0; fi
    echo "[$(date)] waiting siml-09 gpu0 for SAFREE multi VLM eval; mem=$mem util=$util" >&2
    sleep 120
  done
}
run_eval(){
  local dir=$1 concept=$2 name=$3
  local res="$dir/results_qwen3_vl_${concept}_v5.txt"
  [ "$concept" = "nudity" ] && res="$dir/results_qwen3_vl_nudity_v5.txt"
  if [ -s "$res" ]; then echo "[SKIP done] $name"; return 0; fi
  wait_gpu0
  echo "[$(date)] eval $name dir=$dir concept=$concept"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$V5" "$dir" "$concept" qwen 2>&1 | tee "$LOGDIR/${name}.log"
}
run_eval "$BASE/2c_sexvio__safree__eval_sexual" nudity 2c_sexual
run_eval "$BASE/2c_sexvio__safree__eval_violence" violence 2c_violence
run_eval "$BASE/3c_sexvioshock__safree__eval_sexual" nudity 3c_sexual
run_eval "$BASE/3c_sexvioshock__safree__eval_violence" violence 3c_violence
run_eval "$BASE/3c_sexvioshock__safree__eval_shocking" shocking 3c_shocking
run_eval "$BASE/7c_all__safree__eval_sexual" nudity 7c_sexual
run_eval "$BASE/7c_all__safree__eval_violence" violence 7c_violence
run_eval "$BASE/7c_all__safree__eval_self-harm" self_harm 7c_self_harm
run_eval "$BASE/7c_all__safree__eval_shocking" shocking 7c_shocking
run_eval "$BASE/7c_all__safree__eval_illegal_activity" illegal 7c_illegal
run_eval "$BASE/7c_all__safree__eval_harassment" harassment 7c_harassment
run_eval "$BASE/7c_all__safree__eval_hate" hate 7c_hate
echo "[$(date)] SAFREE multi VLM eval finished"
