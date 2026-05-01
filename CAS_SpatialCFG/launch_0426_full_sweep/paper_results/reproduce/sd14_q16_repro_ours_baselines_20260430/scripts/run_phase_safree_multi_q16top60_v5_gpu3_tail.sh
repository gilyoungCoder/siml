#!/usr/bin/env bash
set -uo pipefail
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_safree_multi_q16top60
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
GPU=3
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/phase_safree_multi_q16top60_v5_0501_direct_gpu3
mkdir -p "$LOGDIR"
run_eval(){
  local rel=$1 concept=$2 name=$3
  local dir="$BASE/$rel"
  local res="$dir/results_qwen3_vl_${concept}_v5.txt"
  [ "$concept" = "nudity" ] && res="$dir/results_qwen3_vl_nudity_v5.txt"
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  echo "[$(date)] START $name concept=$concept n=$n dir=$dir"
  if [ -s "$res" ]; then echo "[SKIP existing] $res"; return 0; fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$V5" "$dir" "$concept" qwen 2>&1 | tee "$LOGDIR/${name}.log"
}
# split tail jobs to GPU3; GPU6 is already running the full list and will skip finished files later
run_eval 3c_sexvioshock__safree__eval_shocking shocking 3c_shocking
run_eval 7c_all__safree__eval_sexual nudity 7c_sexual
run_eval 7c_all__safree__eval_violence violence 7c_violence
run_eval 7c_all__safree__eval_self-harm self_harm 7c_self_harm
run_eval 7c_all__safree__eval_shocking shocking 7c_shocking
run_eval 7c_all__safree__eval_illegal_activity illegal 7c_illegal
run_eval 7c_all__safree__eval_harassment harassment 7c_harassment
run_eval 7c_all__safree__eval_hate hate 7c_hate
echo "[$(date)] GPU3 tail done"
