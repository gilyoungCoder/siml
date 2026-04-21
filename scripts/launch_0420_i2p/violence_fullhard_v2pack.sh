#!/bin/bash
# Re-run violence fullhard with concepts_v2/violent pack + top configs.
# Usage: bash violence_fullhard_v2pack.sh <gpu> <slot> <n_slots>
set -uo pipefail
GPU=$1; SLOT=$2; N_SLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/violent/clip_grouped.pt
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_hard/violence_hard.txt
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_fullhard/ours_sd14_v2pack/violence

# 5 configs (top from family RAB + ainp variants)
CONFIGS=(
  "hybrid:20:0.4"
  "hybrid:15:0.5"
  "hybrid:15:0.4"
  "hybrid:10:0.4"
  "anchor:1.0:0.4"
)

N=${#CONFIGS[@]}
for ((i=SLOT; i<N; i+=N_SLOTS)); do
  IFS=: read -r HOW SS IMG <<< "${CONFIGS[$i]}"
  case $HOW in anchor) HOW_ARG=anchor_inpaint ;; *) HOW_ARG=hybrid ;; esac
  CFG="${HOW}_ss${SS}_thr0.1_imgthr${IMG}_both"
  OUT=$OUT_BASE/$CFG
  mkdir -p $OUT
  N_IMGS=$(ls -1 $OUT/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge 313 ]; then echo "[GPU $GPU][skip] $CFG"; continue; fi
  echo "[GPU $GPU][run] $CFG"
  cd $REPO/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
    --prompts $PROMPTS --outdir $OUT \
    --probe_mode both --cas_threshold 0.6 \
    --safety_scale $SS --attn_threshold 0.1 --img_attn_threshold $IMG \
    --how_mode $HOW_ARG --family_guidance --family_config $PACK \
    >> $LOGDIR/violence_fullhard_${CFG}_g${GPU}.log 2>&1
done
