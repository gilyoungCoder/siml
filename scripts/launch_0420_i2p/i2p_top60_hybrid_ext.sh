#!/bin/bash
# I2P top60 — HYBRID-ONLY extended grid for all 6 concepts (higher ss).
# Usage: bash i2p_top60_hybrid_ext.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_grid_v1pack_hyb_ext
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR $OUT_BASE

declare -A TGT_KW ANC_KW PACK_NAME
TGT_KW[violence]="violence bloody_scene weapon fight"
ANC_KW[violence]="peaceful_scene calm_interaction"
PACK_NAME[violence]="violence"
TGT_KW[self-harm]="self_harm cutting pills_suicide noose"
ANC_KW[self-harm]="wellness recovery supportive_therapy"
PACK_NAME[self-harm]="self-harm"
TGT_KW[shocking]="gore body_horror mutilation corpse"
ANC_KW[shocking]="peaceful_scene beautiful_art"
PACK_NAME[shocking]="shocking"
TGT_KW[illegal_activity]="drugs crime contraband illegal_activity"
ANC_KW[illegal_activity]="legal_activity daily_life"
PACK_NAME[illegal_activity]="illegal_activity"
TGT_KW[harassment]="bullying mockery intimidation abuse"
ANC_KW[harassment]="friendly_interaction cooperation"
PACK_NAME[harassment]="harassment"
TGT_KW[hate]="hate_symbol discrimination racist_imagery"
ANC_KW[hate]="diversity respectful_portrait"
PACK_NAME[hate]="hate"

# Hybrid configs only. Higher ss, fill missing img_thr values.
CONFIGS=(
  "25|0.1|0.3"
  "25|0.1|0.5"
  "30|0.1|0.3"
  "30|0.1|0.4"
  "30|0.1|0.5"
  "40|0.1|0.4"
  "40|0.1|0.3"
)

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

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
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  tgt="${TGT_KW[$CONCEPT]}"
  anc="${ANC_KW[$CONCEPT]}"
  outdir="$OUT_BASE/${CONCEPT}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/hyb_ext_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $CONCEPT ss=$SS img=$IMG ($n_imgs)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $CONCEPT hybrid ss=$SS txt=$TXT img=$IMG -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts $tgt --anchor_concepts $anc \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $CONCEPT ss=$SS img=$IMG" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
