#!/bin/bash
# Phase 2 dispatcher: multi-concept sweep (1c/2c/3c/7c × 3 configs each).
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase2_multi
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/phase2_g${GPU}_s${SLOT}.log
echo "[$(date)] [phase2 g$GPU s$SLOT] worker started" | tee -a $LOG
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name setup cfg eval_c n_c packs ss tau thr_t thr_i prompts; do
  outdir=$OUTBASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date)] [phase2 g$GPU s$SLOT] MISSING prompts file: $prompts_abs" | tee -a $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [phase2 g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs" | tee -a $LOG
    continue
  fi
  # Convert relative pack paths to absolute (space-separated list)
  packs_abs=""
  for p in $packs; do
    packs_abs="$packs_abs $REPO/CAS_SpatialCFG/$p"
  done
  mkdir -p "$outdir"
  echo "[$(date)] [phase2 g$GPU s$SLOT] [run  $name] setup=$setup cfg=$cfg eval=$eval_c n_concepts=$n_c n_prompts=$prompt_count" | tee -a $LOG
  cd $REPO/SafeGen
  # Note: cas_threshold / safety_scale / attn_threshold / img_attn_threshold are space-separated lists for nargs="+"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family_multi \
    --prompts "$prompts_abs" \
    --outdir "$outdir" \
    --family_guidance \
    --family_config $packs_abs \
    --probe_mode both --how_mode hybrid \
    --cas_threshold $tau \
    --safety_scale $ss \
    --attn_threshold $thr_t \
    --img_attn_threshold $thr_i \
    >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [phase2 g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [phase2 g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [phase2 g$GPU s$SLOT] worker done" | tee -a $LOG
