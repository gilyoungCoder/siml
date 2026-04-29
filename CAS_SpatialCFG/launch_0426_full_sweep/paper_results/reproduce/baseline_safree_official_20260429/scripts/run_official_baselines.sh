#!/bin/bash
# Sequential official-repo reproduction for SAFREE+SafeDenoiser and SAFREE+SGF on siml-09.
# Usage: bash run_official_baselines.sh [GPU=0] [MODE=full|smoke]
set -uo pipefail
GPU=${1:-0}
MODE=${2:-full}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/baseline_safree_official_20260429
REPO=/mnt/home3/yhgil99/unlearning
SD_REPO=$ROOT/official_repos/Safe_Denoiser
SGF_REPO=$ROOT/official_repos/SGF/nudity_sdv1
OUT=$ROOT/outputs
LOGDIR=$ROOT/logs
MANIFEST=$ROOT/manifest.csv
PY=${PY:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
VLM_PY=${VLM_PY:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
VLM=$REPO/vlm/opensource_vlm_i2p_all_v5.py
mkdir -p "$OUT" "$LOGDIR"
STATUS=$LOGDIR/status_${MODE}.log
log(){ echo "[$(date)] $*" | tee -a "$STATUS"; }

if [ "$MODE" = smoke ]; then RANGE="0,1"; SUFFIX="smoke"; else RANGE="0,100000"; SUFFIX="full"; fi

run_vlm_eval(){
  local imgdir=$1 eval=$2 log=$3
  local n=$(find "$imgdir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then log "WARN no pngs for VLM eval: $imgdir"; return 0; fi
  if ls "$imgdir"/results_qwen3_vl_${eval}_v5.txt >/dev/null 2>&1; then
    log "skip VLM existing $imgdir $eval"
  else
    log "VLM eval $imgdir concept=$eval n=$n"
    (cd "$REPO/vlm" && CUDA_VISIBLE_DEVICES=$GPU "$VLM_PY" "$VLM" "$imgdir" "$eval" qwen >> "$log" 2>&1)
  fi
}

run_job(){
  local method=$1 group=$2 name=$3 data=$4 category=$5 eval=$6 expected=$7
  local repo script erase task_config outdir logf all_dir runner
  if [ "$method" = safedenoiser ]; then
    repo=$SD_REPO
    if [ "$group" = i2p60 ]; then script=run_copro.py; task_config=configs/nudity/safe_denoiser.yaml; else script=run_nudity.py; task_config=configs/nudity/safe_denoiser.yaml; fi
    erase=safree_neg_prompt_rep_threshold_time
  elif [ "$method" = sgf ]; then
    repo=$SGF_REPO
    script=generate_unsafe_sgf.py
    task_config=configs/sgf/sgf.yaml
    erase=safree_neg_prompt_rep_time
  else
    log "unknown method $method"; return 2
  fi
  outdir=$OUT/$SUFFIX/$method/$group/$name
  logf=$LOGDIR/${SUFFIX}_${method}_${group}_${name}.log
  all_dir=$outdir/all
  local have=$(find "$all_dir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$MODE" = full ] && [ "$have" -ge "$expected" ]; then
    log "SKIP gen $method/$group/$name have=$have expected=$expected"
  elif [ "$MODE" = smoke ] && [ "$have" -ge 1 ]; then
    log "SKIP smoke gen $method/$group/$name have=$have"
  else
    rm -rf "$outdir"
    log "GEN $method/$group/$name category=$category range=$RANGE expected=$expected data=$data"
    (cd "$repo" && CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$script" \
      --nudenet-path=pretrained/classifier_model.onnx \
      --nudity_thr=0.6 \
      --num_inference_steps=50 \
      --config=configs/base/vanilla/safree_neg_prompt_config.json \
      --safe_level=MEDIUM \
      --data="$data" \
      --category="$category" \
      --task_config="$task_config" \
      --save-dir="$outdir" \
      --erase_id="$erase" \
      --guidance_scale=7.5 \
      --seed=42 \
      --valid_case_numbers="$RANGE" \
      >> "$logf" 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then log "FAIL gen $method/$group/$name rc=$rc log=$logf"; return $rc; fi
  fi
  run_vlm_eval "$all_dir" "$eval" "$logf"
}

log "START official baseline reproduction MODE=$MODE GPU=$GPU PY=$PY PYTHONNOUSERSITE_GEN=1"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$SD_REPO/run_nudity.py" "$SD_REPO/run_copro.py" "$SGF_REPO/generate_unsafe_sgf.py" || exit 2
# CSV columns: group,name,data,category,eval,expected
{
  read -r header
  while IFS=, read -r group name data category eval expected; do
    [ -z "$group" ] && continue
    # full order: smaller nudity and i2p first; MMA last because it is long.
    if [ "$MODE" = full ] && [ "$name" = mma ]; then continue; fi
    run_job safedenoiser "$group" "$name" "$data" "$category" "$eval" "$expected" || true
    run_job sgf "$group" "$name" "$data" "$category" "$eval" "$expected" || true
  done
  if [ "$MODE" = full ]; then
    grep ',mma,' "$MANIFEST" | while IFS=, read -r group name data category eval expected; do
      run_job safedenoiser "$group" "$name" "$data" "$category" "$eval" "$expected" || true
      run_job sgf "$group" "$name" "$data" "$category" "$eval" "$expected" || true
    done
  fi
} < "$MANIFEST"
"$ROOT/scripts/collect_official_results.py" "$OUT/$SUFFIX" > "$LOGDIR/collect_${SUFFIX}.log" 2>&1 || true
log "DONE official baseline reproduction MODE=$MODE"
