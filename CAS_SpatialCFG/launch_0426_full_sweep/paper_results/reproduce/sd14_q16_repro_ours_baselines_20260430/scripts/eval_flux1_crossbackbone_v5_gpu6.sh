#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
VLM=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
GPU=${GPU:-6}
run_eval(){
  local dir="$1" concept="$2" label="$3"
  [ -d "$dir" ] || { echo "[MISS DIR] $label $dir"; return 0; }
  local n=$(find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l)
  if [ "$n" -eq 0 ]; then echo "[MISS IMG] $label"; return 0; fi
  local result
  case "$concept" in
    sexual|nudity) result=results_qwen3_vl_nudity_v5.txt; concept_arg=nudity;;
    violent|violence) result=results_qwen3_vl_violence_v5.txt; concept_arg=violence;;
    self-harm|self_harm) result=results_qwen3_vl_self_harm_v5.txt; concept_arg=self_harm;;
    illegal|illegal_activity) result=results_qwen3_vl_illegal_v5.txt; concept_arg=illegal;;
    *) result=results_qwen3_vl_${concept}_v5.txt; concept_arg=$concept;;
  esac
  if [ -f "$dir/$result" ]; then echo "[SKIP] $label has $result"; return 0; fi
  echo "[EVAL] $label n=$n concept=$concept_arg"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$VLM" "$dir" "$concept_arg" qwen
}
for method in safree safedenoiser sgf; do
  for c in sexual violence self-harm shocking illegal_activity harassment hate; do
    run_eval "$ROOT/outputs/crossbackbone_0501/flux1/$method/i2p_q16/$c/all" "$c" "flux1/$method/i2p/$c"
  done
  run_eval "$ROOT/outputs/crossbackbone_0501/flux1/$method/mja/mja_sexual/all" sexual "flux1/$method/mja/sexual"
  run_eval "$ROOT/outputs/crossbackbone_0501/flux1/$method/mja/mja_violent/all" violence "flux1/$method/mja/violent"
  run_eval "$ROOT/outputs/crossbackbone_0501/flux1/$method/mja/mja_illegal/all" illegal "flux1/$method/mja/illegal"
  run_eval "$ROOT/outputs/crossbackbone_0501/flux1/$method/mja/mja_disturbing/all" disturbing "flux1/$method/mja/disturbing"
done
echo "[DONE] flux1 crossbackbone v5 eval"
