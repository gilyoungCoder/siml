#!/bin/bash
# SafeDenoiser/SGF SD1.4 multi-concept dispatcher on q16_top60 prompts.
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SGFDIR=$REPO/SGF/nudity_sdv1
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase_safedenoiser_sgf_multi_q16top60
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/safedenoiser_sgf_multi_q16top60_g${GPU}_s${SLOT}.log
echo "[$(date)] [sd_sgf g$GPU s$SLOT] worker started" | tee -a $LOG
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name method combo eval_c csv yaml; do
  outdir=$OUTBASE/$name
  csv_abs=$SGFDIR/$csv
  yaml_abs=$SGFDIR/$yaml
  if [ ! -f "$csv_abs" ]; then
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] MISSING csv: $csv_abs" | tee -a $LOG; continue
  fi
  if [ ! -f "$yaml_abs" ]; then
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] MISSING yaml: $yaml_abs" | tee -a $LOG; continue
  fi
  prompt_count=$(($(wc -l < "$csv_abs") - 1))
  existing=$(ls "$outdir/all"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"/{safe,unsafe,all,ref}
  echo "[$(date)] [sd_sgf g$GPU s$SLOT] [run $name] method=$method combo=$combo eval=$eval_c n=$prompt_count" | tee -a $LOG
  cd $SGFDIR
  if [ "$method" = "safe_denoiser" ]; then
    PY_SCRIPT=generate_unsafe_safedenoiser.py
  elif [ "$method" = "sgf" ]; then
    PY_SCRIPT=generate_unsafe_sgf.py
  else
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] UNKNOWN method=$method" | tee -a $LOG; continue
  fi
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/mnt/home3/yhgil99/unlearning/_stubs $PYTHON $PY_SCRIPT \
    --config $SGFDIR/configs/base/vanilla/std_rep_config.json \
    --task_config "$yaml_abs" \
    --data "$csv_abs" \
    --save-dir "$outdir" \
    --erase_id std_rep \
    --category all \
    --device cuda:0 \
    >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir/all"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [sd_sgf g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [sd_sgf g$GPU s$SLOT] worker done" | tee -a $LOG
