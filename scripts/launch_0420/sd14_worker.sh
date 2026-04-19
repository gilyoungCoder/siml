#!/bin/bash
# SD1.4 ours sweep worker. Usage: bash sd14_worker.sh <gpu_id>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

# Concept-specific target/anchor keywords
declare -A TGT
TGT[sexual]="nudity nude_person naked_body"
TGT[violent]="violence bloody_scene weapon fight"
TGT[disturbing]="grotesque body_horror gore monster"
TGT[illegal]="drugs crime contraband illegal_activity"

declare -A ANC
ANC[sexual]="clothed_person fully_dressed_person"
ANC[violent]="peaceful_scene calm_interaction"
ANC[disturbing]="beautiful_art peaceful_scene"
ANC[illegal]="legal_activity daily_life"

# Job list
JOBS=()
SS_LIST=(0.5 1.0 1.5 2.0 2.5 3.0)
THR_RAB=(0.05 0.1 0.2)
THR_MJA=(0.1 0.2)
HOW_LIST=(anchor_inpaint hybrid)

# RAB (concept=sexual)
for ss in "${SS_LIST[@]}"; do
  for thr in "${THR_RAB[@]}"; do
    for how in "${HOW_LIST[@]}"; do
      JOBS+=("rab|sexual|nudity-ring-a-bell.csv|$ss|$thr|$how")
    done
  done
done
# MJA
for d_c in "mja_sexual|sexual" "mja_violent|violent" "mja_disturbing|disturbing" "mja_illegal|illegal"; do
  d=${d_c%|*}; c=${d_c#*|}
  for ss in "${SS_LIST[@]}"; do
    for thr in "${THR_MJA[@]}"; do
      for how in "${HOW_LIST[@]}"; do
        JOBS+=("$d|$c|$d.txt|$ss|$thr|$how")
      done
    done
  done
done

N=${#JOBS[@]}
echo "[GPU $GPU] Total $N jobs in pool, this worker handles indices $GPU,$((GPU+8)),$((GPU+16))..."

cd $REPO/SafeGen

for ((i=GPU; i<N; i+=8)); do
  job=${JOBS[$i]}
  IFS='|' read -r DSET CONCEPT FNAME SS THR HOW <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$FNAME"
  CONFIG_PATH="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/$CONCEPT/clip_grouped.pt"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14/$DSET/cas0.6_ss${SS}_thr${THR}_${HOW}_both"

  # idempotency: skip if enough images
  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$DSET" = "rab" ]; then EXPECTED=79; else EXPECTED=100; fi
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU] [skip] $DSET ss=$SS thr=$THR $HOW ($N_IMGS imgs already)"
    continue
  fi

  mkdir -p "$OUTDIR"
  TGT_KW=${TGT[$CONCEPT]}
  ANC_KW=${ANC[$CONCEPT]}

  echo "[GPU $GPU] [run $((i/8+1))] $DSET ss=$SS thr=$THR $HOW concept=$CONCEPT"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --family_guidance \
    --family_config "$CONFIG_PATH" \
    --probe_mode both \
    --how_mode "$HOW" \
    --cas_threshold 0.6 \
    --safety_scale "$SS" \
    --attn_threshold "$THR" \
    --img_attn_threshold "$THR" \
    --target_concepts $TGT_KW \
    --anchor_concepts $ANC_KW \
    >> "$LOGDIR/sd14_g${GPU}_pool.log" 2>&1
done
echo "[GPU $GPU] Worker done at $(date)"
