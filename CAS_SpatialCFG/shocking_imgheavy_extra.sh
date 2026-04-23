#!/bin/bash
set -uo pipefail
SLOT=$1
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60/shocking_sweep.txt
FAM=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0423_shocking_imgheavy
LOGD=$OUT_BASE/_gen_logs
mkdir -p $LOGD

# slot 0..3 -> ss=22 jobs 20..23
JOBS=("22|0.25|0.1" "22|0.25|0.15" "22|0.3|0.1" "22|0.3|0.15")
IFS='|' read -r ss txt img <<< "${JOBS[$SLOT]}"
TAG=hybrid_ss${ss}_thr${txt}_imgthr${img}_both
OUTDIR=$OUT_BASE/$TAG

LOG=$LOGD/extra_$SLOT.log
echo "[extra-$SLOT] start host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES tag=$TAG" >> $LOG

if [ -f "$OUTDIR/generation_stats.json" ]; then
  echo "[extra-$SLOT] SKIP (already done)" >> $LOG
  exit 0
fi

cd $REPO/SafeGen
$PY -m safegen.generate_family \
  --prompts $PROMPT --outdir $OUTDIR \
  --how_mode hybrid --probe_mode both \
  --safety_scale $ss --attn_threshold $txt --img_attn_threshold $img \
  --cas_threshold 0.6 \
  --family_config $FAM --family_guidance \
  --target_words gore body horror mutilation corpse \
  --target_concepts gore body_horror mutilation corpse \
  --n_img_tokens 4 \
  >> $LOG 2>&1
echo "[extra-$SLOT] DONE" >> $LOG
