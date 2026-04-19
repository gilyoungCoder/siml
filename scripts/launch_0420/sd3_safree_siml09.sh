#!/bin/bash
# SD3 SAFREE worker for siml-09 GPU 0
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

JOBS=(
  "rab|sexual|nudity-ring-a-bell.csv"
  "mja_sexual|sexual|mja_sexual.txt"
  "mja_violent|violence|mja_violent.txt"
  "mja_disturbing|shocking|mja_disturbing.txt"
  "mja_illegal|illegal|mja_illegal.txt"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r DSET CONCEPT FNAME <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$FNAME"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/safree_sd3/$DSET"

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$DSET" = "rab" ]; then EXPECTED=79; else EXPECTED=100; fi
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[skip] SD3 SAFREE $DSET ($N_IMGS imgs)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[run] SD3 SAFREE $DSET concept=$CONCEPT"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/scripts/sd3/generate_sd3_safree.py \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --concept "$CONCEPT" \
    --device cuda:0 \
    --no_cpu_offload \
    >> "$LOGDIR/sd3_safree_siml09_g${GPU}.log" 2>&1
done
echo "[GPU $GPU] SD3 SAFREE worker done at $(date)"
