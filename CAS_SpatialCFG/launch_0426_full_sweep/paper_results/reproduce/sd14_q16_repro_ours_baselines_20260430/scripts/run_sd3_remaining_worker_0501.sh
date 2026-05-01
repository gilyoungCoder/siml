#!/usr/bin/env bash
set -uo pipefail
SHARD=${1:?shard}; NSHARDS=${2:?nshards}; GPU=${3:?gpu}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
WRAP=$ROOT/scripts/run_sd3_with_sfgd_torch_safree_libs.py
GEN=$ROOT/scripts/generate_sd3_prompt_repellency.py
JOBLIST=$ROOT/joblists/sd3_remaining_official_0501.tsv
count_pngs(){ find "$1/all" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l; }
idx=0; TAB=$(printf '\t')
while IFS="$TAB" read -r backbone method dataset concept prompt out expected task; do
  [ -z "${backbone:-}" ] && continue
  if [ $((idx % NSHARDS)) -ne "$SHARD" ]; then idx=$((idx+1)); continue; fi
  echo "========== SD3 REMAIN idx=$idx shard=$SHARD/$NSHARDS host=$(hostname) gpu=$GPU $method $dataset $concept =========="
  mkdir -p "$out"
  n=$(count_pngs "$out")
  if [ "$n" -ge "$expected" ]; then echo "[SKIP] already $n/$expected"; idx=$((idx+1)); continue; fi
  # keep partials for debug, but remove before rerun only if very low/incomplete
  rm -rf "$out"; mkdir -p "$out"
  extra=(); [ "$task" != "NONE" ] && extra=(--task_config "$task")
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$WRAP" "$GEN" \
    --method "$method" --prompts "$prompt" --outdir "$out" "${extra[@]}" \
    --steps 28 --guidance_scale 7.0 --height 1024 --width 1024 --device cuda:0
  rc=$?
  echo "========== END idx=$idx rc=$rc count=$(count_pngs "$out")/$expected =========="
  idx=$((idx+1))
done < "$JOBLIST"
