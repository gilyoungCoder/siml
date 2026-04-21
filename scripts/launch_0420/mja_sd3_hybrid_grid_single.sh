#!/bin/bash
# SD3 MJA hybrid grid for a SINGLE dataset (sexual/violent/illegal).
# Usage: bash mja_sd3_hybrid_grid_single.sh <gpu_id> <dataset>
#        dataset ∈ {mja_sexual, mja_violent, mja_illegal}
set -uo pipefail
GPU=$1
DS=$2
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR
LOG="$LOGDIR/sd3_${DS}_hybrid_g${GPU}.log"

declare -A PROMPTS PACK
PROMPTS[mja_sexual]="$REPO/CAS_SpatialCFG/prompts/mja_sexual.txt"
PROMPTS[mja_violent]="$REPO/CAS_SpatialCFG/prompts/mja_violent.txt"
PROMPTS[mja_illegal]="$REPO/CAS_SpatialCFG/prompts/mja_illegal.txt"
PACK[mja_sexual]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
PACK[mja_violent]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/violent/clip_grouped.pt"
PACK[mja_illegal]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/illegal/clip_grouped.pt"

prompts="${PROMPTS[$DS]}"
pack="${PACK[$DS]}"

JOBS=(
  "10|0.1|0.3"  "10|0.1|0.4"  "10|0.1|0.5"
  "15|0.1|0.3"  "15|0.1|0.4"  "15|0.1|0.5"
  "20|0.1|0.3"  "20|0.1|0.4"  "20|0.1|0.5"
)

for cfg in "${JOBS[@]}"; do
  IFS='|' read -r SS TXT IMG <<< "$cfg"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/${DS}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 100 ]; then
    echo "[$(date)] [g$GPU] [skip] $DS ss=$SS img=$IMG ($n_imgs/100)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $DS ss=$SS img=$IMG -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$REPO/scripts/sd3/generate_sd3_safegen.py" \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $DS ss=$SS img=$IMG" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU] done for $DS"
