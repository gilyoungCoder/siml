#!/bin/bash
# I2P top60 ablation (txt-only / img-only) with REPATCHED i2p_v1 packs.
# 3 cfgs per probe per concept = 36 jobs total.
# Usage: bash i2p_ablation_repatched.sh <gpu> <slot> <nshards>
set -uo pipefail
GPU=$1; SLOT=$2; NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

declare -A PACK
PACK[violence]="violence"
PACK[self-harm]="self-harm"
PACK[shocking]="shocking"
PACK[illegal_activity]="illegal_activity"
PACK[harassment]="harassment"
PACK[hate]="hate"

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

# 3 cfgs per probe (ss=15 fixed, vary thresholds)
TXT_CFGS=("15|0.1|0.3" "15|0.2|0.3" "15|0.3|0.3")  # vary txt_thr
IMG_CFGS=("15|0.1|0.3" "15|0.1|0.4" "15|0.1|0.5")  # vary img_thr

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${TXT_CFGS[@]}"; do JOBS+=("txt|$c|$cfg"); done
  for cfg in "${IMG_CFGS[@]}"; do JOBS+=("img|$c|$cfg"); done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSHARDS)); do
  IFS='|' read -r PROBE CONCEPT SS TXT IMG <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK[$CONCEPT]}/clip_grouped.pt"
  if [ "$PROBE" = "txt" ]; then
    pmode="text"; outdir_dir="ours_sd14_ablation_txtonly_repatched"
    sub="hybrid_ss${SS}_thr${TXT}_txtonly"
  else
    pmode="image"; outdir_dir="ours_sd14_ablation_imgonly_repatched"
    sub="hybrid_ss${SS}_imgthr${IMG}_imgonly"
  fi
  outdir="$OUT_BASE/$outdir_dir/$CONCEPT/$sub"
  LOG="$LOGDIR/abl_repat_g${GPU}_s${SLOT}.log"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 60 ] && { echo "[skip] $PROBE $CONCEPT ss=$SS ($n)" | tee -a "$LOG"; continue; }
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $PROBE $CONCEPT ss=$SS txt=$TXT img=$IMG" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode "$pmode" --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    >> "$LOG" 2>&1 || echo "FAILED $PROBE $CONCEPT" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
