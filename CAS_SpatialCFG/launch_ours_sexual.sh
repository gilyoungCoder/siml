#!/bin/bash
# usage: launch_ours_sexual.sh <gpu> <sd3|flux1>
set -uo pipefail
G="$1"; BB="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

if [ "$BB" = "sd3" ]; then
  SS=20.0; TAU=0.5; PACK_DIR=i2p_v1_sd3
  GEN=$REPO/scripts/sd3/generate_sd3_safegen.py
  EXTRA=""
else
  SS=2.0; TAU=0.5; PACK_DIR=i2p_v1_flux1
  GEN=$REPO/CAS_SpatialCFG/generate_flux1_v1.py
  EXTRA="--dtype bfloat16"
fi
PACK=$REPO/CAS_SpatialCFG/exemplars/$PACK_DIR/sexual/clip_grouped.pt
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/sexual_q16_top60.txt
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_${BB}/sexual/hybrid_ss${SS}_thr0.15_imgthr0.1_cas${TAU}_both
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_${BB}/_logs
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
readarray -t TW < <($PY -c "
import torch
pack = torch.load('$PACK', map_location='cpu', weights_only=False)
fm = pack['family_metadata']
tw = []
for f in fm:
    for ph in fm[f].get('target_words', []):
        for w in ph.replace('_',' ').split():
            wl = w.strip().lower()
            if len(wl) >= 3 and wl not in tw: tw.append(wl)
print(chr(10).join(tw))
")

CMD="setsid env CUDA_VISIBLE_DEVICES=$G $PY $GEN \
  --prompts $PROMPTS --outdir $OUT \
  --family_config $PACK --family_guidance \
  --probe_mode both --how_mode hybrid \
  --safety_scale $SS --attn_threshold 0.15 --img_attn_threshold 0.10 \
  --cas_threshold $TAU --n_img_tokens 4 \
  --target_concepts ${TC[@]@Q} --anchor_concepts ${AC[@]@Q} \
  $EXTRA"

if [ "$BB" = "sd3" ]; then
  CMD="$CMD --target_words ${TW[@]@Q}"
fi

eval "$CMD </dev/null > $LOGD/g${G}_sexual_ours_hyb.log 2>&1 &"
echo "started ours $BB sexual on g$G pid=$!"
