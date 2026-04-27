#!/bin/bash
# SAFREE v2 sweep on siml-09 GPU 0 — paper-aligned: --safree -svf -lra
# Sequential, 22 cells (16 multi + 6 i2p single).
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=$REPO/.conda/envs/sdd_copy/bin/python3.10
[ -x /mnt/home3/yhgil99/.conda/envs/sdd/bin/python3.10 ] && PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd/bin/python3.10

TSV=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/cells_safree_v2.tsv
OUT_BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_safree_v2
LOGDIR=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/logs
mkdir -p $OUT_BASE $LOGDIR

LOG=$LOGDIR/safree_v2.log
echo "[$(date)] SAFREE v2 sweep START (--safree -svf -lra)" | tee $LOG

tail -n +2 $TSV | while IFS=$'\t' read -r name setup eval_c n_c cats_csv prompts; do
  outdir=$OUT_BASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date +%H:%M:%S)] MISSING prompts: $prompts_abs" >> $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name ($existing/$prompt_count)" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date +%H:%M:%S)] RUN $name setup=$setup eval=$eval_c category=$cats_csv n_prompts=$prompt_count" | tee -a $LOG
  cd $REPO/SAFREE
  CUDA_VISIBLE_DEVICES=0 $PYTHON gen_safree_single.py \
    --txt "$prompts_abs" \
    --save-dir "$outdir" \
    --category "$cats_csv" \
    --re_attn_t=-1,1001 \
    --linear_per_prompt_seed \
    --num_inference_steps 50 \
    --safree -svf -lra \
    >> $LOG 2>&1
  rc=$?
  # SAFREE writes to outdir/generated/, move up to outdir/
  mv $outdir/generated/*.png $outdir/ 2>/dev/null || true
  rmdir $outdir/generated 2>/dev/null || true
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAIL $name exit=$rc" | tee -a $LOG
  else
    final=$(ls $outdir/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] DONE $name $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] SAFREE v2 sweep DONE" | tee -a $LOG
