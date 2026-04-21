#!/bin/bash
# SD1.4 MJA hybrid grid for sexual/violent/illegal (3 datasets × 9 cfgs = 27 jobs).
# Usage: bash mja_sd14_hybrid_grid_svi.sh <gpu_id> <slot_idx> <n_slots>
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
PACK[mja_sexual]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
PACK[mja_violent]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/violent/clip_grouped.pt"
PACK[mja_illegal]="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/illegal/clip_grouped.pt"
TGT_KW[mja_sexual]="nudity nude_person naked_body"
ANC_KW[mja_sexual]="clothed_person fully_dressed_person"
TGT_KW[mja_violent]="violence bloody_scene weapon fight"
ANC_KW[mja_violent]="peaceful_scene calm_interaction"
TGT_KW[mja_illegal]="drugs crime contraband illegal_activity"
ANC_KW[mja_illegal]="legal_activity daily_life"

CONFIGS=(
  "10|0.1|0.3"  "10|0.1|0.4"  "10|0.1|0.5"
  "15|0.1|0.3"  "15|0.1|0.4"  "15|0.1|0.5"
  "20|0.1|0.3"  "20|0.1|0.4"  "20|0.1|0.5"
)

JOBS=()
for ds in mja_sexual mja_violent mja_illegal; do
  for cfg in "${CONFIGS[@]}"; do
    JOBS+=("$ds|$cfg")
  done
done
N=${#JOBS[@]}

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r DS SS TXT IMG <<< "${JOBS[$i]}"
  prompts="${PROMPTS[$DS]}"
  pack="${PACK[$DS]}"
  tgt="${TGT_KW[$DS]}"
  anc="${ANC_KW[$DS]}"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14/${DS}/hybrid_ss${SS}_thr${TXT}_imgthr${IMG}_both"
  LOG="$LOGDIR/sd14_svi_hybrid_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 100 ]; then
    echo "[$(date)] [g$GPU] [skip] $DS ss=$SS img=$IMG ($n_imgs/100)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [run] $DS ss=$SS img=$IMG -> $outdir" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$pack" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts $tgt --anchor_concepts $anc \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $DS ss=$SS img=$IMG" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
