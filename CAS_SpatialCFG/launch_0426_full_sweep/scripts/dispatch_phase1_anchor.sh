#!/bin/bash
# Phase 1B dispatcher: single-concept ANCHOR mode reproducibility.
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
# TSV columns: cell_name mode pack ss tau thr_t thr_i probe_mode prompts
# Uses generate_family.py with --how_mode anchor_inpaint.
# Resume: counts existing PNGs and passes --start_idx to skip already-done prompts.
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase1b_anchor
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/phase1b_g${GPU}_s${SLOT}.log
echo "[$(date)] [phase1b g$GPU s$SLOT] worker started" | tee -a $LOG
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name mode pack ss tau thr_t thr_i probe_mode prompts; do
  outdir=$OUTBASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  pack_abs=$REPO/CAS_SpatialCFG/$pack/clip_grouped.pt
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date)] [phase1b g$GPU s$SLOT] MISSING prompts file: $prompts_abs" | tee -a $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [phase1b g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [phase1b g$GPU s$SLOT] [run  $name] mode=$mode pack=$pack ss=$ss tau=$tau thr_t=$thr_t thr_i=$thr_i probe=$probe_mode start_idx=$existing n_prompts=$prompt_count" | tee -a $LOG
  cd $REPO/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts_abs" \
    --outdir "$outdir" \
    --start_idx "$existing" \
    --family_guidance \
    --family_config "$pack_abs" \
    --probe_mode "$probe_mode" --how_mode "$mode" \
    --cas_threshold "$tau" --safety_scale "$ss" \
    --attn_threshold "$thr_t" --img_attn_threshold "$thr_i" \
    >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [phase1b g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [phase1b g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [phase1b g$GPU s$SLOT] worker done" | tee -a $LOG
