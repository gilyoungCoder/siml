#!/bin/bash
# Phase 1 dispatcher: single-concept reproducibility (paper Table 8 hyperparams).
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
# 16 workers (2 per GPU on siml-01); slot index 0..15; gpu = slot // 2.
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase1_single
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/phase1_g${GPU}_s${SLOT}.log
echo "[$(date)] [phase1 g$GPU s$SLOT] worker started" | tee -a $LOG
# Read TSV (skip header), dispatch by (rownum-2) % NSLOTS == SLOT
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name mode pack ss tau thr_t thr_i prompts target_override; do
  outdir=$OUTBASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  pack_abs=$REPO/CAS_SpatialCFG/$pack/clip_grouped.pt
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date)] [phase1 g$GPU s$SLOT] MISSING prompts file: $prompts_abs" | tee -a $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [phase1 g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [phase1 g$GPU s$SLOT] [run  $name] mode=$mode pack=$pack ss=$ss tau=$tau thr_t=$thr_t thr_i=$thr_i n_prompts=$prompt_count" | tee -a $LOG
  cd $REPO/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts_abs" \
    --outdir "$outdir" \
    --family_guidance \
    --family_config "$pack_abs" \
    --probe_mode both --how_mode "$mode" \
    --cas_threshold "$tau" --safety_scale "$ss" \
    --attn_threshold "$thr_t" --img_attn_threshold "$thr_i" \
    >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [phase1 g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [phase1 g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [phase1 g$GPU s$SLOT] worker done" | tee -a $LOG
