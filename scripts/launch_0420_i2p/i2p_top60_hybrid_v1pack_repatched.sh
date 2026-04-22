#!/bin/bash
# I2P top60 hybrid sweep with REPATCHED i2p_v1 packs (family-specific keywords).
# Best config range: hybrid + both, ss∈{10,15,20,25} × imgthr∈{0.3,0.4,0.5}.
# Usage: bash i2p_top60_hybrid_v1pack_repatched.sh <gpu> <slot> <nshards>
set -uo pipefail
GPU=$1; SLOT=$2; NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_v1pack_repatched
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR $OUT_BASE

declare -A PACK_NAME
PACK_NAME[violence]="violence"
PACK_NAME[self-harm]="self-harm"
PACK_NAME[shocking]="shocking"
PACK_NAME[illegal_activity]="illegal_activity"
PACK_NAME[harassment]="harassment"
PACK_NAME[hate]="hate"

CONFIGS=(
  "10|0.1|0.3"  "10|0.1|0.4"  "10|0.1|0.5"
  "15|0.1|0.3"  "15|0.1|0.4"  "15|0.1|0.5"
  "20|0.1|0.3"  "20|0.1|0.4"  "20|0.1|0.5"
  "25|0.1|0.3"  "25|0.1|0.4"  "25|0.1|0.5"
)
CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do JOBS+=("$c|$cfg"); done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSHARDS)); do
  IFS='|' read -r CONCEPT SS TXT IMG <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  outdir="$OUT_BASE/${CONCEPT}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/repatched_g${GPU}_s${SLOT}.log"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 60 ] && { echo "[skip] $CONCEPT ss=$SS img=$IMG ($n)" | tee -a "$LOG"; continue; }
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $CONCEPT hybrid ss=$SS img=$IMG" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    >> "$LOG" 2>&1 || echo "FAILED $CONCEPT ss=$SS" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
