#!/bin/bash
set -uo pipefail
WORKER=$1
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0423_harhate_imgheavy
LOGD=$OUT_BASE/_gen_logs
mkdir -p $LOGD

# 24 jobs: 2 concepts × 2 ss × 3 txt × 2 img
JOBS=()
for cdir in harassment hate; do
  for ss in 20 22; do
    for txt in 0.15 0.2 0.25; do
      for img in 0.1 0.15; do
        JOBS+=("$cdir|$ss|$txt|$img")
      done
    done
  done
done
N=${#JOBS[@]}
WC=23
echo "[w$WORKER] start host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES total=$N" >> $LOGD/worker_$WORKER.log

cd $REPO/SafeGen

for ((i=WORKER; i<N; i+=WC)); do
  IFS='|' read -r cdir ss txt img <<< "${JOBS[$i]}"
  PROMPT=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60/${cdir}_sweep.txt
  FAM=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/${cdir}/clip_grouped.pt
  TAG=${cdir}/hybrid_ss${ss}_thr${txt}_imgthr${img}_both
  OUTDIR=$OUT_BASE/$TAG
  if [ -f "$OUTDIR/generation_stats.json" ]; then
    echo "[w$WORKER] SKIP $TAG" >> $LOGD/worker_$WORKER.log
    continue
  fi
  echo "[w$WORKER] GEN $TAG" >> $LOGD/worker_$WORKER.log
  if [ "$cdir" = harassment ]; then
    TW='nudity nude person naked body'
    TC='nudity nude_person naked_body'
  else
    TW='hate symbol discrimination racist imagery'
    TC='hate_symbol discrimination racist_imagery'
  fi
  $PY -m safegen.generate_family \
    --prompts $PROMPT --outdir $OUTDIR \
    --how_mode hybrid --probe_mode both \
    --safety_scale $ss --attn_threshold $txt --img_attn_threshold $img \
    --cas_threshold 0.6 \
    --family_config $FAM --family_guidance \
    --target_words $TW --target_concepts $TC \
    --n_img_tokens 4 \
    >> $LOGD/worker_$WORKER.log 2>&1
done
echo "[w$WORKER] DONE" >> $LOGD/worker_$WORKER.log
