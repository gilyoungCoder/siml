#!/bin/bash
# MJA-disturbing SD1.4 hybrid ours — LOW imgthr extension (ss∈{10,15,20} × imgthr∈{0.1, 0.2}).
# Usage: bash mja_disturbing_hybrid_grid_sd14_ext.sh <gpu_id> <slot_idx> <n_slots>
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
OUT_BASE="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14/mja_disturbing"

JOBS=(
  "10|0.1|0.1"  "10|0.1|0.2"
  "15|0.1|0.1"  "15|0.1|0.2"
  "20|0.1|0.1"  "20|0.1|0.2"
)
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r SS TXT_THR IMG_THR <<< "${JOBS[$i]}"
  outdir="$OUT_BASE/hybrid_ss${SS}_thr${TXT_THR}_imgthr${IMG_THR}_both_retune"
  LOG="$LOGDIR/mja_disturbing_sd14_ext_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 100 ]; then
    echo "[$(date)] [g$GPU] [skip] ss=$SS imgthr=$IMG_THR" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] ss=$SS txt=$TXT_THR img=$IMG_THR -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" --outdir "$outdir" \
    --family_guidance --family_config "$PACK" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT_THR" --img_attn_threshold "$IMG_THR" \
    --target_concepts grotesque body_horror gore monster \
    --anchor_concepts beautiful_art peaceful_scene \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED ss=$SS imgthr=$IMG_THR" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
