#!/bin/bash
# I2P top60 anchor_inpaint both probe sweep — ss >= 1.5.
# Usage: bash i2p_top60_anchor_sweep.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_anchor_sweep
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

CONFIGS=(
  "1.2|0.1|0.3"  "1.2|0.1|0.4"  "1.2|0.1|0.5"
  "1.5|0.1|0.3"  "1.5|0.1|0.4"  "1.5|0.1|0.5"
  "1.8|0.1|0.3"  "1.8|0.1|0.4"  "1.8|0.1|0.5"
  "2.0|0.1|0.3"  "2.0|0.1|0.4"  "2.0|0.1|0.5"
  "2.5|0.1|0.3"  "2.5|0.1|0.4"  "2.5|0.1|0.5"
)
CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do JOBS+=("$c|$cfg"); done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r CONCEPT SS TXT IMG <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  outdir="$OUT_BASE/${CONCEPT}/anchor_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/anchor_sweep_g${GPU}_s${SLOT}.log"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 60 ] && { echo "[skip] $CONCEPT ss=$SS img=$IMG ($n)" | tee -a "$LOG"; continue; }
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $CONCEPT anchor ss=$SS img=$IMG" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode anchor_inpaint \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts ${TGT_KW[$CONCEPT]} --anchor_concepts ${ANC_KW[$CONCEPT]} \
    >> "$LOG" 2>&1 || echo "FAILED $CONCEPT ss=$SS" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
