#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
VLM=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
GPU=${GPU:-0}
run_eval(){
  local dir="$1" concept="$2" name="$3"
  local expected
  case "$concept" in
    nudity) expected=results_qwen3_vl_nudity_v5.txt;;
    self-harm|self_harm) expected=results_qwen3_vl_self_harm_v5.txt;;
    illegal|illegal_activity) expected=results_qwen3_vl_illegal_v5.txt;;
    *) expected=results_qwen3_vl_${concept}_v5.txt;;
  esac
  expected=${expected//-/_}
  if [ -f "$dir/$expected" ]; then
    echo "[SKIP] $name already has $expected"
    return 0
  fi
  echo "[EVAL] $name concept=$concept dir=$dir"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$VLM" "$dir" "$concept" qwen
}
run_eval "$ROOT/outputs/safedenoiser_multi_7c/i2p_q16/hate/all" hate safedenoiser_multi_7c_hate
run_eval "$ROOT/outputs/sgf_multi_7c/i2p_q16/harassment/all" harassment sgf_multi_7c_harassment
run_eval "$ROOT/outputs/sgf_multi_7c/i2p_q16/illegal_activity/all" illegal sgf_multi_7c_illegal_activity
echo "[DONE] missing multi v5 eval"
