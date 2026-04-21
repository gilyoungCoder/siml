#!/bin/bash
# I2P top60 ours grid-B — complementary configs (anchor_inpaint + ss=12,18,22) for siml-06/09 workers.
# 10 configs × 6 I2P concepts = 60 jobs. Writes to ours_sd14_grid_v1pack_b/ (separate from grid-A to avoid races).
# Usage: bash i2p_top60_ours_grid_b.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_grid_v1pack_b
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

# Complementary grid: off-center ss values (12, 18, 22) + anchor_inpaint variants.
CONFIGS=(
  "hybrid|12|0.1|0.4"
  "hybrid|18|0.1|0.4"
  "hybrid|22|0.1|0.4"
  "hybrid|12|0.1|0.5"
  "hybrid|18|0.1|0.5"
  "anchor_inpaint|1.0|0.1|0.4"
  "anchor_inpaint|1.5|0.1|0.4"
  "anchor_inpaint|2.0|0.1|0.4"
  "anchor_inpaint|1.5|0.1|0.5"
  "anchor_inpaint|2.0|0.1|0.3"
)

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    JOBS+=("$c|$cfg")
  done
done
N=${#JOBS[@]}

echo "[$(date)] [gridB g$GPU s$SLOT/$NSLOTS] $N total jobs, my slice = $SLOT,$(($SLOT+$NSLOTS)),..."

cd $REPO/SafeGen

for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r CONCEPT HOW SS TXT_THR IMG_THR <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  tgt="${TGT_KW[$CONCEPT]}"
  anc="${ANC_KW[$CONCEPT]}"
  outdir="$OUT_BASE/${CONCEPT}/${HOW}_ss${SS}_thr${TXT_THR}_imgthr${IMG_THR}_both"
  LOG="$LOGDIR/ours_gridB_g${GPU}_s${SLOT}.log"

  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU gridB] [skip] $CONCEPT $HOW ss=$SS ($n_imgs imgs)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  if [ ! -f "$pack" ]; then
    echo "[g$GPU gridB] [missing pack] $pack" | tee -a "$LOG"
    continue
  fi
  echo "[$(date)] [g$GPU gridB] [run] $CONCEPT $HOW ss=$SS txt=$TXT_THR img=$IMG_THR -> $outdir" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" \
    --outdir "$outdir" \
    --family_guidance \
    --family_config "$pack" \
    --probe_mode both \
    --how_mode "$HOW" \
    --cas_threshold 0.6 \
    --safety_scale "$SS" \
    --attn_threshold "$TXT_THR" \
    --img_attn_threshold "$IMG_THR" \
    --target_concepts $tgt \
    --anchor_concepts $anc \
    >> "$LOG" 2>&1 || echo "[g$GPU gridB] FAILED $CONCEPT $HOW ss=$SS" | tee -a "$LOG"
done

echo "[$(date)] [g$GPU gridB s$SLOT] done"
