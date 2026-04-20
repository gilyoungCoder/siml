#!/bin/bash
# Focused family RAB sweep — explore hybrid + ainp around v27 best (94.9% baseline)
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR
PROMPTS=$REPO/CAS_SpatialCFG/prompts/ringabell.txt
PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/family_rab

JOBS=()
# Hybrid sweep (v27 best=15, ts=as)
for ss in 10 15 20 25; do
  for img in 0.3 0.4 0.5; do
    JOBS+=("hybrid|$ss|0.1|$img|both")
  done
done
# Anchor inpaint sweep (ss small)
for ss in 0.8 1.0 1.2 1.5; do
  for img in 0.3 0.4; do
    JOBS+=("anchor|$ss|0.1|$img|both")
  done
done
# Probe ablation at hybrid ss=15 best
for pb in imgonly txtonly; do
  JOBS+=("hybrid|15|0.1|0.4|$pb")
  JOBS+=("anchor|1.0|0.1|0.4|$pb")
done

N=${#JOBS[@]}
echo "[GPU $GPU] family RAB sweep: $N configs"

cd $REPO/SafeGen
for job in "${JOBS[@]}"; do
  IFS='|' read -r HOW SS THR IMG_THR PB <<< "$job"
  CFG="${HOW}_ss${SS}_thr${THR}_imgthr${IMG_THR}_${PB}"
  OUTDIR="$OUT_BASE/$CFG"
  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge 79 ]; then
    echo "[GPU $GPU][skip] $CFG ($N_IMGS imgs)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[GPU $GPU][run] $CFG"
  case $PB in
    imgonly) PB_ARG=image ;;
    txtonly) PB_ARG=text ;;
    *) PB_ARG=both ;;
  esac
  case $HOW in
    anchor) HOW_ARG=anchor_inpaint ;;
    *) HOW_ARG=hybrid ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" --outdir "$OUTDIR" \
    --probe_mode $PB_ARG --cas_threshold 0.6 \
    --safety_scale $SS --attn_threshold $THR --img_attn_threshold $IMG_THR \
    --how_mode $HOW_ARG --family_guidance --family_config "$PACK" \
    >> "$LOGDIR/family_rab_${CFG}_g${GPU}.log" 2>&1
done
echo "[GPU $GPU] family RAB sweep done at $(date)"
