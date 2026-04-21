#!/bin/bash
# Ours multi-concept erasure on I2P top60 — hybrid sweep.
# Loads all 6 I2P family packs (sexual/violence/self-harm/shocking/illegal/harassment/hate)
# and performs flat winner-take-all across all 6×4=24 families at once.
# Usage: bash ours_multi_i2p_top60.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_multi
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR $OUT_BASE

PACKS=(
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/illegal_activity/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/harassment/clip_grouped.pt"
  "$REPO/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt"
)

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)
CONFIGS=(
  "10|0.1|0.3"
  "15|0.1|0.3"
  "15|0.1|0.4"
  "20|0.1|0.3"
  "20|0.1|0.4"
)

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    JOBS+=("$c|$cfg")
  done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r CONCEPT SS TXT IMG <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  outdir="$OUT_BASE/${CONCEPT}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/ours_multi_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $CONCEPT ss=$SS img=$IMG ($n_imgs)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] multi $CONCEPT ss=$SS txt=$TXT img=$IMG -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family_multi \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "${PACKS[@]}" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $CONCEPT ss=$SS img=$IMG" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
