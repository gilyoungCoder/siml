#!/bin/bash
# Extended grid for harassment / self-harm / hate on I2P top60 — push SR higher.
# More aggressive ss, different cas_threshold, anchor variants.
# Usage: bash i2p_top60_hard_concepts_extended.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_grid_v1pack_ext
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR $OUT_BASE

declare -A TGT_KW ANC_KW PACK_NAME
TGT_KW[self-harm]="self_harm cutting pills_suicide noose"
ANC_KW[self-harm]="wellness recovery supportive_therapy"
PACK_NAME[self-harm]="self-harm"
TGT_KW[harassment]="bullying mockery intimidation abuse"
ANC_KW[harassment]="friendly_interaction cooperation"
PACK_NAME[harassment]="harassment"
TGT_KW[hate]="hate_symbol discrimination racist_imagery"
ANC_KW[hate]="diversity respectful_portrait"
PACK_NAME[hate]="hate"

# Each cfg: CAS|HOW|SS|TXT|IMG
CONFIGS=(
  "0.6|hybrid|25|0.1|0.4"
  "0.6|hybrid|30|0.1|0.4"
  "0.6|hybrid|40|0.1|0.4"
  "0.6|anchor_inpaint|3.0|0.1|0.3"
  "0.6|anchor_inpaint|3.0|0.1|0.4"
  "0.5|hybrid|20|0.1|0.4"
  "0.7|hybrid|20|0.1|0.4"
  "0.6|hybrid|15|0.05|0.2"
)

CONCEPTS=(self-harm harassment hate)

JOBS=()
for c in "${CONCEPTS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    JOBS+=("$c|$cfg")
  done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r CONCEPT CAS HOW SS TXT IMG <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  tgt="${TGT_KW[$CONCEPT]}"
  anc="${ANC_KW[$CONCEPT]}"
  outdir="$OUT_BASE/${CONCEPT}/${HOW}_ss${SS}_thr${TXT}_imgthr${IMG}_cas${CAS}_both"
  LOG="$LOGDIR/hard_ext_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $CONCEPT ($n_imgs imgs)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $CONCEPT cas=$CAS $HOW ss=$SS txt=$TXT img=$IMG -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode "$HOW" \
    --cas_threshold "$CAS" --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts $tgt --anchor_concepts $anc \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $CONCEPT" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
