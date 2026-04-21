#!/bin/bash
# SD1.4 baseline generation on I2P Q16-top60 subsets.
# Usage: bash i2p_q16top60_baseline_worker.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_q16top60/baseline_sd14
LOGDIR=$REPO/logs/launch_0420_i2p_q16top60
mkdir -p $LOGDIR

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)
N=${#CONCEPTS[@]}

for ((i=SLOT; i<N; i+=NSLOTS)); do
  concept=${CONCEPTS[$i]}
  prompts="$PROMPT_DIR/${concept}_q16_top60.txt"
  outdir="$OUT_BASE/$concept"

  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $concept ($n_imgs imgs)"
    continue
  fi

  mkdir -p "$outdir"
  LOG="$LOGDIR/baseline_${concept}_g${GPU}.log"
  echo "[$(date)] [g$GPU] gen baseline $concept -> $outdir" | tee "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$REPO/CAS_SpatialCFG/generate_baseline.py" \
    --prompts "$prompts" \
    --outdir "$outdir" \
    --nsamples 1 \
    --seed 42 \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $concept" | tee -a "$LOG"
done

echo "[$(date)] [g$GPU slot $SLOT] done"
