#!/bin/bash
# FLUX1 mja_violent + mja_illegal hybrid grid (ss∈{15,20} × imgthr∈{0.3,0.4,0.5}).
# Usage: bash mja_flux1_hybrid_violent_illegal.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

declare -A PROMPTS PACK
PROMPTS[mja_violent]="$REPO/CAS_SpatialCFG/prompts/mja_violent.txt"
PROMPTS[mja_illegal]="$REPO/CAS_SpatialCFG/prompts/mja_illegal.txt"
PACK[mja_violent]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/violent/clip_grouped.pt"
PACK[mja_illegal]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/illegal/clip_grouped.pt"

CONFIGS=(
  "15|0.1|0.3"  "15|0.1|0.4"  "15|0.1|0.5"
  "20|0.1|0.3"  "20|0.1|0.4"  "20|0.1|0.5"
)

JOBS=()
for ds in mja_violent mja_illegal; do
  for cfg in "${CONFIGS[@]}"; do
    JOBS+=("$ds|$cfg")
  done
done
N=${#JOBS[@]}

for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r DS SS TXT IMG <<< "${JOBS[$i]}"
  prompts="${PROMPTS[$DS]}"
  pack="${PACK[$DS]}"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/${DS}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/flux1_hyb_vi_g${GPU}_s${SLOT}.log"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n" -ge 100 ]; then
    echo "[skip] $DS ss=$SS img=$IMG ($n)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $DS ss=$SS img=$IMG" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$REPO/CAS_SpatialCFG/generate_flux1_v1.py" \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --device cuda:0 --height 1024 --width 1024 \
    --start_idx "$n" \
    >> "$LOG" 2>&1 || echo "FAILED $DS ss=$SS img=$IMG" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
