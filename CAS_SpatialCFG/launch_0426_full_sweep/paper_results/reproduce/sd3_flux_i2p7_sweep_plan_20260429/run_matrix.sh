#!/bin/bash
# Ours sweep worker for SD3 / FLUX.1-dev I2P-7.
# Usage: bash run_matrix.sh <sd3|flux1> <slot_idx> <n_slots> <gpu_id> [pilot|full]
# Example: bash run_matrix.sh flux1 0 8 0 pilot
set -uo pipefail
MODEL=${1:?model sd3|flux1}
SLOT=${2:?slot_idx}
NSLOTS=${3:?n_slots}
GPU=${4:?gpu_id}
MODE=${5:-pilot}

REPO=/mnt/home3/yhgil99/unlearning
PLAN_ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd3_flux_i2p7_sweep_plan_20260429
OUT_ROOT=${OUT_ROOT:-/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_sd3_flux_i2p7_sweep}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
PY_GEN=${PY_GEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PY_VLM=${PY_VLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
EVAL_SCRIPT=$REPO/vlm/opensource_vlm_i2p_all_v5.py
MATRIX=$PLAN_ROOT/matrix_${MODEL}.tsv
mkdir -p "$LOG_ROOT" "$OUT_ROOT"

case "$MODE" in
  pilot) START_IDX=0; END_IDX=${END_IDX_OVERRIDE:-12}; EXPECTED=$END_IDX ;;
  full) START_IDX=0; END_IDX=${END_IDX_OVERRIDE:-60}; EXPECTED=$END_IDX ;;
  *) echo "MODE must be pilot or full" >&2; exit 2 ;;
esac

if [ ! -f "$MATRIX" ]; then echo "Missing matrix: $MATRIX" >&2; exit 2; fi
LOG=$LOG_ROOT/${MODEL}_${MODE}_g${GPU}_s${SLOT}.log
echo "[$(date)] start MODEL=$MODEL MODE=$MODE SLOT=$SLOT/$NSLOTS GPU=$GPU MATRIX=$MATRIX" | tee -a "$LOG"

idx=-1
tail -n +2 "$MATRIX" | while IFS=$'\t' read -r concept config_id how cas ss tt ia ntok eval_concept prompt_file family_config target_pipe anchor_pipe rationale; do
  idx=$((idx+1))
  if [ $((idx % NSLOTS)) -ne "$SLOT" ]; then continue; fi
  outdir="$OUT_ROOT/$MODEL/$MODE/$concept/$config_id"
  mkdir -p "$outdir"
  n_png=$(find "$outdir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "[$(date)] [$MODEL/$MODE g$GPU] job#$idx concept=$concept cfg=$config_id imgs=$n_png/$EXPECTED why=$rationale" | tee -a "$LOG"

  if [ "$n_png" -lt "$EXPECTED" ]; then
    IFS='|' read -r -a targets <<< "$target_pipe"
    IFS='|' read -r -a anchors <<< "$anchor_pipe"
    if [ "$MODEL" = "sd3" ]; then
      CUDA_VISIBLE_DEVICES=$GPU "$PY_GEN" "$REPO/scripts/sd3/generate_sd3_safegen.py" \
        --prompts "$prompt_file" --outdir "$outdir" \
        --family_guidance --family_config "$family_config" \
        --probe_mode both --how_mode "$how" \
        --cas_threshold "$cas" --safety_scale "$ss" \
        --attn_threshold "$tt" --img_attn_threshold "$ia" --n_img_tokens "$ntok" \
        --steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42 \
        --start_idx "$START_IDX" --end_idx "$END_IDX" \
        --target_concepts "${targets[@]}" --anchor_concepts "${anchors[@]}" \
        >> "$LOG" 2>&1
    elif [ "$MODEL" = "flux1" ]; then
      CUDA_VISIBLE_DEVICES=$GPU "$PY_GEN" "$REPO/CAS_SpatialCFG/generate_flux1_v1.py" \
        --prompts "$prompt_file" --outdir "$outdir" \
        --family_guidance --family_config "$family_config" \
        --probe_mode both --how_mode "$how" \
        --cas_threshold "$cas" --safety_scale "$ss" \
        --attn_threshold "$tt" --img_attn_threshold "$ia" --n_img_tokens "$ntok" \
        --steps 28 --guidance_scale 3.5 --height 512 --width 512 --seed 42 \
        --start_idx "$START_IDX" --end_idx "$END_IDX" --device cuda:0 \
        --target_concepts "${targets[@]}" --anchor_concepts "${anchors[@]}" \
        >> "$LOG" 2>&1
    else
      echo "Unknown MODEL=$MODEL" >&2; exit 2
    fi
  else
    echo "[$(date)] skip generation existing $n_png >= $EXPECTED" | tee -a "$LOG"
  fi

  eval_marker="$outdir/.eval_v5_qwen3_vl_${eval_concept}.done"
  if [ ! -f "$eval_marker" ]; then
    cd "$REPO/vlm" || exit 1
    CUDA_VISIBLE_DEVICES=$GPU "$PY_VLM" "$EVAL_SCRIPT" "$outdir" "$eval_concept" qwen >> "$LOG" 2>&1 && touch "$eval_marker"
    cd "$REPO" || exit 1
  else
    echo "[$(date)] skip eval marker exists $eval_marker" | tee -a "$LOG"
  fi
done

echo "[$(date)] done MODEL=$MODEL MODE=$MODE SLOT=$SLOT/$NSLOTS GPU=$GPU" | tee -a "$LOG"
