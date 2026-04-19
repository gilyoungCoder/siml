#!/bin/bash
# SD3 ours sweep worker (3-block hooks). Waits for GPU free (<10GB) then runs.
# Usage: bash sd3_ours_worker_v2.sh <gpu_id> <slot 0..3>
set -uo pipefail
GPU=$1
SLOT=$2
N_SLOTS=4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR

# Wait until GPU free (used_memory < 10GB)
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  if [ "$used" -lt 10000 ]; then
    echo "[GPU $GPU] free ($used MB), starting"
    break
  fi
  echo "[GPU $GPU] waiting ($used MB used > 10GB threshold)"
  sleep 60
done

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

JOBS=()
SS_LIST=(0.5 1.0 1.5 2.0 2.5 3.0)
THR_RAB=(0.05 0.1 0.2)
THR_MJA=(0.1 0.2)
HOW_LIST=(anchor_inpaint hybrid)
for ss in "${SS_LIST[@]}"; do
  for thr in "${THR_RAB[@]}"; do
    for how in "${HOW_LIST[@]}"; do
      JOBS+=("rab|sexual|nudity-ring-a-bell.csv|$ss|$thr|$how")
    done
  done
done
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
echo "[GPU $GPU/slot $SLOT] $N total jobs, slice from index $SLOT step $N_SLOTS"

cd $REPO/scripts/sd3

for ((i=SLOT; i<N; i+=N_SLOTS)); do
  job=${JOBS[$i]}
  IFS='|' read -r DSET CONCEPT FNAME SS THR HOW <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$FNAME"
  CONFIG_PATH="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/$CONCEPT/clip_grouped.pt"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/$DSET/cas0.6_ss${SS}_thr${THR}_${HOW}_both"

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$DSET" = "rab" ]; then EXPECTED=79; else EXPECTED=100; fi
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU] [skip] $DSET ss=$SS thr=$THR $HOW ($N_IMGS imgs)"
    continue
  fi
  mkdir -p "$OUTDIR"
  TGT_KW=${TGT[$CONCEPT]}
  ANC_KW=${ANC[$CONCEPT]}

  echo "[GPU $GPU] [run $((i/N_SLOTS+1))] $DSET ss=$SS thr=$THR $HOW concept=$CONCEPT"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/scripts/sd3/generate_sd3_safegen.py \
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
    >> "$LOGDIR/sd3_ours_v2_g${GPU}_pool.log" 2>&1
done
echo "[GPU $GPU/slot $SLOT] Worker done at $(date)"
