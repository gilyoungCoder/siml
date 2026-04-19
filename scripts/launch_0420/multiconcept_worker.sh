#!/bin/bash
# Multi-concept (sexual+violent) test worker for SD1.4 on siml-01 GPU 6
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
PACK=$REPO/CAS_SpatialCFG/exemplars/multiconcept_v1/sexual+violent/clip_grouped.pt
TGT="nudity nude_person naked_body violence bloody_scene weapon fight"
ANC="clothed_person peaceful_scene calm_interaction"
mkdir -p $LOGDIR

# Sweep configs
JOBS=()
for dset in mja_sexual mja_violent; do
  for ss in 0.5 1.0 1.5 2.0 2.5 3.0; do
    for thr in 0.1 0.2; do
      for how in anchor_inpaint hybrid; do
        JOBS+=("$dset|$ss|$thr|$how")
      done
    done
  done
done

N=${#JOBS[@]}
echo "[GPU $GPU] $N multiconcept jobs to run"

cd $REPO/SafeGen

for i in "${!JOBS[@]}"; do
  job=${JOBS[$i]}
  IFS='|' read -r DSET SS THR HOW <<< "$job"
  PROMPTS="$REPO/CAS_SpatialCFG/prompts/$DSET.txt"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14_multiconcept/$DSET/cas0.6_ss${SS}_thr${THR}_${HOW}_both"

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge 100 ]; then
    echo "[skip] $DSET ss=$SS thr=$THR $HOW ($N_IMGS imgs)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[run $((i+1))/$N] multiconcept $DSET ss=$SS thr=$THR $HOW"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --family_guidance \
    --family_config "$PACK" \
    --probe_mode both \
    --how_mode "$HOW" \
    --cas_threshold 0.6 \
    --safety_scale "$SS" \
    --attn_threshold "$THR" \
    --img_attn_threshold "$THR" \
    --target_concepts $TGT \
    --anchor_concepts $ANC \
    >> "$LOGDIR/multiconcept_g${GPU}_pool.log" 2>&1
done
echo "[GPU $GPU] multiconcept worker done at $(date)"
