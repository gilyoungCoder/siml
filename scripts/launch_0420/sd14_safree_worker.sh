#!/bin/bash
# SD1.4 SAFREE worker. Usage: bash sd14_safree_worker.sh <gpu_id> <slot 0..1>
set -uo pipefail
GPU=$1
SLOT=$2
N_SLOTS=2
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

JOBS=(
  "rab|nudity-ring-a-bell.csv|csv"
  "mja_sexual|mja_sexual.txt|txt"
  "mja_violent|mja_violent.txt|txt"
  "mja_disturbing|mja_disturbing.txt|txt"
  "mja_illegal|mja_illegal.txt|txt"
)

N=${#JOBS[@]}
echo "[GPU $GPU/slot $SLOT] $N SD1.4 SAFREE jobs, slice from $SLOT step $N_SLOTS"

cd $REPO/SAFREE

for ((i=SLOT; i<N; i+=N_SLOTS)); do
  job=${JOBS[$i]}
  IFS='|' read -r DSET FNAME KIND <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$FNAME"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/safree_sd14/$DSET"

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$DSET" = "rab" ]; then EXPECTED=79; else EXPECTED=100; fi
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU] [skip] SAFREE $DSET ($N_IMGS imgs)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[GPU $GPU] [run] SAFREE $DSET"

  if [ "$KIND" = "csv" ]; then
    DATA_ARG="--csv $PROMPTS --csv-col prompt"
  else
    DATA_ARG="--txt $PROMPTS"
  fi

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/SAFREE/gen_safree_simple.py \
    --safree --svf --lra \
    $DATA_ARG \
    --outdir "$OUTDIR" \
    --num_images 1 --steps 50 --seed 42 \
    --height 512 --width 512 \
    >> "$LOGDIR/sd14_safree_g${GPU}.log" 2>&1
done
echo "[GPU $GPU/slot $SLOT] SD1.4 SAFREE worker done at $(date)"
