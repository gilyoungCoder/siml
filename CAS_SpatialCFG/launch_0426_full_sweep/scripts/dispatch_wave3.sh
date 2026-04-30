#!/bin/bash
# Wave3 dispatcher: SD3 SafeDenoiser + SGF on q16_top60 (7 concepts each).
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
CASDIR=$REPO/CAS_SpatialCFG
OUTBASE=$CASDIR/launch_0426_full_sweep/outputs/wave3_sd3_q16top60
LOGDIR=$CASDIR/launch_0426_full_sweep/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/wave3_g${GPU}_s${SLOT}.log
echo "[$(date)] [wave3 g$GPU s$SLOT] worker started" | tee -a $LOG

awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name method concept csv yaml; do
  outdir=$OUTBASE/$name
  prompt_count=$(($(wc -l < "$csv_abs" 2>/dev/null) - 1))
  if [ "$method" = "safedenoiser" ]; then
    SDV3DIR=$REPO/Safe_Denoiser
    PY_SCRIPT=run_nudity_sdv3.py
    ERASE_ID=safree_neg_prompt_rep_time
  elif [ "$method" = "sgf" ]; then
    SDV3DIR=$REPO/SGF/diversity_sdv3
    PY_SCRIPT=generate_sdv3.py
    ERASE_ID=sgf
  else
    echo "[$(date)] [wave3 g$GPU s$SLOT] UNKNOWN method=$method" | tee -a $LOG; continue
  fi
  csv_abs=$SDV3DIR/$csv
  yaml_abs=$SDV3DIR/$yaml
  if [ ! -f "$csv_abs" ]; then echo "[$(date)] MISSING csv $csv_abs" | tee -a $LOG; continue; fi
  if [ ! -f "$yaml_abs" ]; then echo "[$(date)] MISSING yaml $yaml_abs" | tee -a $LOG; continue; fi
  prompt_count=$(($(wc -l < "$csv_abs") - 1))
  existing=$(ls "$outdir/all"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [wave3 g$GPU s$SLOT] [skip $name] $existing/$prompt_count (done)" | tee -a $LOG
    continue
  fi
  if [ "$existing" -gt 0 ]; then
    echo "[$(date)] [wave3 g$GPU s$SLOT] [skip $name] $existing/$prompt_count (in-progress)" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"/{safe,unsafe,all,ref}
  echo "[$(date)] [wave3 g$GPU s$SLOT] [run $name] method=$method concept=$concept n=$prompt_count" | tee -a $LOG
  cd $SDV3DIR
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/mnt/home3/yhgil99/unlearning/_stubs $PYTHON $PY_SCRIPT \
    --config $SDV3DIR/configs/base/vanilla/sdv3_config.json \
    --task_config "$yaml_abs" \
    --data "$csv_abs" \
    --save-dir "$outdir" \
    --erase_id $ERASE_ID \
    --category all \
    --device cuda:0 \
    >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [wave3 g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir/all"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [wave3 g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [wave3 g$GPU s$SLOT] worker done" | tee -a $LOG
