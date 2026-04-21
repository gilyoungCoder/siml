#!/bin/bash
# MJA-disturbing FLUX1 hybrid ours grid: ss∈{10,15,20} × imgthr∈{0.3,0.4,0.5}.
# Usage: bash mja_disturbing_hybrid_grid_flux1.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR
PROMPTS="$REPO/CAS_SpatialCFG/prompts/mja_disturbing.txt"
PACK="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/disturbing/clip_grouped.pt"
OUT_BASE="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_flux1/mja_disturbing"

JOBS=(
  "10|0.1|0.3"  "10|0.1|0.4"  "10|0.1|0.5"
  "15|0.1|0.3"  "15|0.1|0.4"  "15|0.1|0.5"
  "20|0.1|0.3"  "20|0.1|0.4"  "20|0.1|0.5"
)
N=${#JOBS[@]}

for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r SS TXT_THR IMG_THR <<< "${JOBS[$i]}"
  outdir="$OUT_BASE/hybrid_ss${SS}_thr${TXT_THR}_imgthr${IMG_THR}_both"
  LOG="$LOGDIR/mja_disturbing_flux1_hybrid_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 100 ]; then
    echo "[$(date)] [g$GPU] [skip] ss=$SS imgthr=$IMG_THR" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] ss=$SS txt=$TXT_THR img=$IMG_THR -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/CAS_SpatialCFG/generate_flux1_v1.py \
    --prompts "$PROMPTS" --outdir "$outdir" \
    --family_guidance --family_config "$PACK" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT_THR" --img_attn_threshold "$IMG_THR" \
    --device cuda:0 --height 1024 --width 1024 \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED ss=$SS imgthr=$IMG_THR" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
