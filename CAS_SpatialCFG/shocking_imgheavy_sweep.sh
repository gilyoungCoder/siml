#!/bin/bash
set -uo pipefail
WORKER=$1
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPT=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60/shocking_sweep.txt
FAM=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0423_shocking_imgheavy
LOGD=$OUT_BASE/_gen_logs
mkdir -p $LOGD

JOBS=()
for ss in 18 20 22; do
  for txt in 0.15 0.2 0.25 0.3; do
    for img in 0.1 0.15; do
      JOBS+=("$ss|$txt|$img")
    done
  done
done

N=${#JOBS[@]}
WC=19
echo "[w$WORKER] start host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES total=$N" >> $LOGD/worker_$WORKER.log

cd $REPO/SafeGen

for ((i=WORKER; i<N; i+=WC)); do
  IFS='|' read -r ss txt img <<< "${JOBS[$i]}"
  TAG=hybrid_ss${ss}_thr${txt}_imgthr${img}_both
  OUTDIR=$OUT_BASE/$TAG
  if [ -f "$OUTDIR/generation_stats.json" ]; then
    echo "[w$WORKER] SKIP $TAG" >> $LOGD/worker_$WORKER.log
    continue
  fi
  echo "[w$WORKER] GEN $TAG" >> $LOGD/worker_$WORKER.log
  $PY -m safegen.generate_family \
    --prompts $PROMPT --outdir $OUTDIR \
    --how_mode hybrid --probe_mode both \
    --safety_scale $ss --attn_threshold $txt --img_attn_threshold $img \
    --cas_threshold 0.6 \
    --family_config $FAM --family_guidance \
    --target_words gore body horror mutilation corpse \
    --target_concepts gore body_horror mutilation corpse \
    --n_img_tokens 4 \
    >> $LOGD/worker_$WORKER.log 2>&1
done
echo "[w$WORKER] DONE" >> $LOGD/worker_$WORKER.log
