#!/bin/bash
# usage: launch_sd14_hyb_ss.sh <gpu> <concept> <ss>
set -uo pipefail
G="$1"; C="$2"; SS="$3"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${C}_q16_top60.txt
PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/$C/clip_grouped.pt
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_ours_hyb_sd14/${C}/hybrid_ss${SS}_thr0.15_imgthr0.10_cas0.6_both
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_ours_hyb_sd14/_logs
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

cd $REPO/SafeGen
setsid env CUDA_VISIBLE_DEVICES=$G $PY -m safegen.generate_family \
  --prompts "$PROMPTS" --outdir "$OUT" \
  --family_config "$PACK" --family_guidance \
  --probe_mode both --how_mode hybrid \
  --safety_scale $SS --attn_threshold 0.15 --img_attn_threshold 0.10 \
  --cas_threshold 0.6 --n_img_tokens 4 \
  --target_concepts "${TC[@]}" --anchor_concepts "${AC[@]}" --target_words "${TW[@]}" \
  </dev/null > "$LOGD/g${G}_${C}_ss${SS}.log" 2>&1 &
echo "started SD14 hyb $C ss=$SS on g$G pid=$!"
