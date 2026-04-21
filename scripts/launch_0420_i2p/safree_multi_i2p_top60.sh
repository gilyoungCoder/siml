#!/bin/bash
# SAFREE multi-concept erasure on I2P top60 (all 7 I2P concepts erased simultaneously).
# Usage: bash safree_multi_i2p_top60.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/safree_sd14_multi
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR $OUT_BASE

MULTI_CATS="nudity,violence,self_harm,shocking,illegal_activity,harassment,hate"
CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

JOBS=("${CONCEPTS[@]}")
N=${#JOBS[@]}

cd $REPO/SAFREE
for ((i=SLOT; i<N; i+=NSLOTS)); do
  CONCEPT=${JOBS[$i]}
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  outdir="$OUT_BASE/${CONCEPT}"
  LOG="$LOGDIR/safree_multi_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $CONCEPT ($n_imgs/60)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] SAFREE multi $CONCEPT -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_safree_single.py \
    --txt "$prompts" \
    --save_dir "$outdir" \
    --category "$MULTI_CATS" \
    --re_attn_t=-1,1001 \
    --linear_per_prompt_seed \
    --num_inference_steps 50 \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $CONCEPT" | tee -a "$LOG"
  mv "$outdir"/generated/*.png "$outdir"/ 2>/dev/null || true
done
echo "[$(date)] [g$GPU s$SLOT] done"
