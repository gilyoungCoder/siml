#!/bin/bash
# Launch one FLUX shard. usage: launch_shard.sh <gpu> <widx> <start> <end>
set -uo pipefail
G="$1"; WIDX="$2"; START="$3"; END="$4"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

CONCEPTS=(violence self-harm shocking illegal_activity harassment hate)
CAT=${CONCEPTS[$WIDX]}
case $CAT in
  illegal_activity) TAU=0.45 ;;
  *) TAU=0.5 ;;
esac

PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1_flux1/$CAT/clip_grouped.pt
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${CAT}_q16_top60.txt
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/$CAT/hybrid_ss2.0_thr0.15_imgthr0.1_cas${TAU}_both
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_flux1/_logs
mkdir -p "$OUT" "$LOGD"

readarray -t TC < <($PY -c "
import torch
pack = torch.load('$PACK', map_location='cpu', weights_only=False)
for f in pack['family_metadata']: print(f.replace('_',' '))
")
readarray -t AC < <($PY -c "
import torch
pack = torch.load('$PACK', map_location='cpu', weights_only=False)
for f, fd in pack['family_metadata'].items(): print(fd.get('anchor_words', ['safe'])[0])
")

setsid env CUDA_VISIBLE_DEVICES=$G $PY $REPO/CAS_SpatialCFG/generate_flux1_v1.py \
  --prompts "$PROMPTS" --outdir "$OUT" \
  --family_config "$PACK" --family_guidance \
  --probe_mode both --how_mode hybrid \
  --safety_scale 2.0 --attn_threshold 0.15 --img_attn_threshold 0.10 \
  --cas_threshold $TAU --n_img_tokens 4 \
  --start_idx $START --end_idx $END \
  --dtype bfloat16 \
  --target_concepts "${TC[@]}" --anchor_concepts "${AC[@]}" \
  </dev/null > "$LOGD/g${G}_w${WIDX}_${START}-${END}.log" 2>&1 &
echo "started shard $CAT [$START,$END) on g$G pid=$!"
