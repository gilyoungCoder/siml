#!/bin/bash
# FLUX1 SAFREE sequential worker for siml-09 GPU 0
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

declare -A TGT
TGT[sexual]="nudity nude_person naked_body"
TGT[violent]="violence bloody_scene weapon fight"
TGT[disturbing]="grotesque body_horror gore monster"
TGT[illegal]="drugs crime contraband illegal_activity"

JOBS=(
  "rab|sexual|nudity-ring-a-bell.csv"
  "mja_sexual|sexual|mja_sexual.txt"
  "mja_violent|violent|mja_violent.txt"
  "mja_disturbing|disturbing|mja_disturbing.txt"
  "mja_illegal|illegal|mja_illegal.txt"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r DSET CONCEPT FNAME <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$FNAME"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/safree_flux1/$DSET"
  TGT_KW=${TGT[$CONCEPT]}

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$DSET" = "rab" ]; then EXPECTED=79; else EXPECTED=100; fi
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[skip] $DSET ($N_IMGS imgs)"
    continue
  fi

  mkdir -p "$OUTDIR"
  echo "[run] FLUX1 SAFREE $DSET concept=$CONCEPT"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/CAS_SpatialCFG/generate_flux1_safree.py \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --target_concepts $TGT_KW \
    --safree_token_filter \
    --safree_re_attention \
    --device cuda:0 \
    >> "$LOGDIR/flux1_safree_g${GPU}_pool.log" 2>&1
done
echo "[GPU $GPU] FLUX1 SAFREE worker done at $(date)"
