#!/bin/bash
# I2P top60 ablation sweep: harassment/hate grid-A catch-up + txt-only + img-only hybrid.
# Usage: bash i2p_top60_ablation_all.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

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

ALL_CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)
HARD_CONCEPTS=(harassment hate)

JOBS=()

# 1) harassment + hate grid-A catch-up (hybrid both, ss∈{10,15,20,25} × img∈{0.3,0.4,0.5})
for c in "${HARD_CONCEPTS[@]}"; do
  for ss in 10 15 20 25; do
    for img in 0.3 0.4 0.5; do
      outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_grid_v1pack/${c}/hybrid_ss${ss}_thr0.1_imgthr${img}_both"
      JOBS+=("both|${c}|${ss}|0.1|${img}|${outdir}")
    done
  done
done

# 2) txt-only hybrid: ss∈{15,20} × thr∈{0.1,0.2,0.3,0.4}, all 6 concepts
for c in "${ALL_CONCEPTS[@]}"; do
  for ss in 15 20; do
    for thr in 0.1 0.2 0.3 0.4; do
      outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_ablation_txtonly/${c}/hybrid_ss${ss}_thr${thr}_txtonly"
      JOBS+=("txtonly|${c}|${ss}|${thr}|0.3|${outdir}")
    done
  done
done

# 3) img-only hybrid: ss∈{15,20} × img_thr∈{0.1,0.2,0.3,0.4}, all 6 concepts (no 0.5 per user)
for c in "${ALL_CONCEPTS[@]}"; do
  for ss in 15 20; do
    for img in 0.1 0.2 0.3 0.4; do
      outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/ours_sd14_ablation_imgonly/${c}/hybrid_ss${ss}_thr0.1_imgthr${img}_imgonly"
      JOBS+=("imgonly|${c}|${ss}|0.1|${img}|${outdir}")
    done
  done
done

N=${#JOBS[@]}
echo "[$(date)] [g$GPU s$SLOT/$NSLOTS] total=$N jobs" | tee -a "$LOGDIR/ablation_all_g${GPU}_s${SLOT}.log"

cd $REPO/SafeGen
for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r PROBE CONCEPT SS TXT IMG OUTDIR <<< "${JOBS[$i]}"
  prompts="$PROMPT_DIR/${CONCEPT}_sweep.txt"
  pack="$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${PACK_NAME[$CONCEPT]}/clip_grouped.pt"
  tgt="${TGT_KW[$CONCEPT]}"
  anc="${ANC_KW[$CONCEPT]}"
  LOG="$LOGDIR/ablation_all_g${GPU}_s${SLOT}.log"
  n_imgs=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge 60 ]; then
    echo "[$(date)] [g$GPU] [skip] $PROBE $CONCEPT ss=$SS img=$IMG ($n_imgs)" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[$(date)] [g$GPU] [run] $PROBE $CONCEPT ss=$SS txt=$TXT img=$IMG -> $OUTDIR" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$OUTDIR" \
    --family_guidance --family_config "$pack" \
    --probe_mode $(case "$PROBE" in txtonly) echo text;; imgonly) echo image;; *) echo both;; esac) --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold "$TXT" --img_attn_threshold "$IMG" \
    --target_concepts $tgt --anchor_concepts $anc \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED $PROBE $CONCEPT ss=$SS" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
