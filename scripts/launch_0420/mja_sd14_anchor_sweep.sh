#!/bin/bash
# MJA SD1.4 anchor_inpaint both probe sweep — ss >= 1.5.
# 4 datasets (sexual/violent/illegal/disturbing) × 12 cfgs.
# Usage: bash mja_sd14_anchor_sweep.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

declare -A PROMPTS PACK TGT_KW ANC_KW
PROMPTS[mja_sexual]="$REPO/CAS_SpatialCFG/prompts/mja_sexual.txt"
PROMPTS[mja_violent]="$REPO/CAS_SpatialCFG/prompts/mja_violent.txt"
PROMPTS[mja_illegal]="$REPO/CAS_SpatialCFG/prompts/mja_illegal.txt"
PROMPTS[mja_disturbing]="$REPO/CAS_SpatialCFG/prompts/mja_disturbing.txt"
PACK[mja_sexual]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
PACK[mja_violent]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/violent/clip_grouped.pt"
PACK[mja_illegal]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/illegal/clip_grouped.pt"
PACK[mja_disturbing]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/disturbing/clip_grouped.pt"
TGT_KW[mja_sexual]="nudity nude_person naked_body"
ANC_KW[mja_sexual]="clothed_person fully_dressed_person"
TGT_KW[mja_violent]="violence bloody_scene weapon fight"
ANC_KW[mja_violent]="peaceful_scene calm_interaction"
TGT_KW[mja_illegal]="drugs crime contraband illegal_activity"
ANC_KW[mja_illegal]="legal_activity daily_life"
TGT_KW[mja_disturbing]="grotesque body_horror gore monster"
ANC_KW[mja_disturbing]="beautiful_art peaceful_scene"

CONFIGS=(
  "1.2|0.1|0.3"  "1.2|0.1|0.4"  "1.2|0.1|0.5"
  "1.5|0.1|0.3"  "1.5|0.1|0.4"  "1.5|0.1|0.5"
  "1.8|0.1|0.3"  "1.8|0.1|0.4"  "1.8|0.1|0.5"
  "2.0|0.1|0.3"  "2.0|0.1|0.4"  "2.0|0.1|0.5"
  "2.5|0.1|0.3"  "2.5|0.1|0.4"  "2.5|0.1|0.5"
)
DSETS=(mja_sexual mja_violent mja_illegal mja_disturbing)

JOBS=()
for d in "${DSETS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do JOBS+=("$d|$cfg"); done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r DS SS TXT IMG <<< "${JOBS[$i]}"
  prompts="${PROMPTS[$DS]}"
  pack="${PACK[$DS]}"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14/${DS}/anchor_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/sd14_anchor_g${GPU}_s${SLOT}.log"
  n=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 100 ] && { echo "[skip] $DS ss=$SS img=$IMG ($n)" | tee -a "$LOG"; continue; }
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $DS anchor ss=$SS img=$IMG" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode anchor_inpaint \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts ${TGT_KW[$DS]} --anchor_concepts ${ANC_KW[$DS]} \
    >> "$LOG" 2>&1 || echo "FAILED $DS ss=$SS" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
