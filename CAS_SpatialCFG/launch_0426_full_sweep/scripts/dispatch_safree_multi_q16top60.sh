#!/bin/bash
# SAFREE multi-concept dispatcher on q16_top60 prompts.
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase_safree_multi_q16top60
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/safree_multi_q16top60_g${GPU}_s${SLOT}.log
echo "[$(date)] [safree_multi g$GPU s$SLOT] worker started" | tee -a $LOG
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name combo cats eval_c prompts; do
  outdir=$OUTBASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date)] [safree_multi g$GPU s$SLOT] MISSING prompts: $prompts_abs" | tee -a $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [safree_multi g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [safree_multi g$GPU s$SLOT] [run $name] cats=$cats eval=$eval_c n_imgs=$prompt_count" | tee -a $LOG
  cd $REPO/SAFREE
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_safree_single.py \
    --txt "$prompts_abs" \
    --save-dir "$outdir" \
    --category "$cats" \
    --safree -svf -lra \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --image_length 512 \
    --seed 42 \
    --linear_per_prompt_seed \
    >> $LOG 2>&1
  rc=$?
  # SAFREE saves under <outdir>/generated/ — move up
  if [ -d "$outdir/generated" ]; then
    mv "$outdir/generated"/*.png "$outdir/" 2>/dev/null || true
    rmdir "$outdir/generated" 2>/dev/null || true
  fi
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [safree_multi g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [safree_multi g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [safree_multi g$GPU s$SLOT] worker done" | tee -a $LOG
