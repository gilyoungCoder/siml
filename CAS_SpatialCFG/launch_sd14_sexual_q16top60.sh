#!/bin/bash
# Launch SD1.4 baseline + SAFREE + ours{hybrid,anchor} on i2p sexual q16_top60.
# Usage: ./launch_sd14_sexual_q16top60.sh <gpu> <method:baseline|safree|ours_hyb|ours_anc>
set -uo pipefail
G="$1"; M="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/sexual_q16_top60.txt
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sexual_sd14/_logs
mkdir -p "$LOGD"

case "$M" in
  baseline)
    OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sexual_sd14/baseline
    mkdir -p "$OUT"
    setsid env CUDA_VISIBLE_DEVICES=$G $PY $REPO/CAS_SpatialCFG/generate_baseline.py \
      --nsamples 1 --prompts "$PROMPTS" --outdir "$OUT" --steps 50 \
      </dev/null > "$LOGD/g${G}_${M}.log" 2>&1 &
    ;;
  safree)
    OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sexual_sd14/safree
    mkdir -p "$OUT"
    cd $REPO/SAFREE
    setsid env CUDA_VISIBLE_DEVICES=$G $PY gen_safree_single.py \
      --txt "$PROMPTS" --save-dir "$OUT" \
      --model_id CompVis/stable-diffusion-v1-4 --category nudity \
      --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
      --seed 42 --image_length 512 --device cuda:0 --erase-id std \
      --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
      --safree -svf -lra --linear_per_prompt_seed \
      </dev/null > "$LOGD/g${G}_${M}.log" 2>&1 &
    ;;
  ours_hyb|ours_anc)
    if [ "$M" = "ours_hyb" ]; then
      MODE=hybrid; SS=22.0; TXT=0.15; IMG=0.10
    else
      MODE=anchor_inpaint; SS=2.0; TXT=0.10; IMG=0.40
    fi
    PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt
    OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_sexual_sd14/ours_${MODE}_ss${SS}_thr${TXT}_imgthr${IMG}_cas0.6_both
    mkdir -p "$OUT"

    # extract TC/AC/TW from pack via rerun_launcher.extract_from_pack
    TC_TXT=$($PY -c "
import torch
pack = torch.load('$PACK', map_location='cpu', weights_only=False)
fm = pack['family_metadata']
tc = [f.replace('_',' ') for f in fm.keys()]
ac = [fm[f].get('anchor_words', ['safe'])[0] for f in fm.keys()]
tw = []
for f in fm:
    for ph in fm[f].get('target_words', []):
        for w in ph.replace('_',' ').split():
            wl = w.strip().lower()
            if len(wl) >= 3 and wl not in tw: tw.append(wl)
print('TC=' + '|'.join(tc))
print('AC=' + '|'.join(ac))
print('TW=' + '|'.join(tw))
")
    TC=($(echo "$TC_TXT" | grep ^TC= | sed 's/TC=//' | tr '|' '\n'))
    AC=($(echo "$TC_TXT" | grep ^AC= | sed 's/AC=//' | tr '|' '\n'))
    TW=($(echo "$TC_TXT" | grep ^TW= | sed 's/TW=//' | tr '|' '\n'))

    cd $REPO/SafeGen
    setsid env CUDA_VISIBLE_DEVICES=$G $PY -m safegen.generate_family \
      --prompts "$PROMPTS" --outdir "$OUT" \
      --family_config "$PACK" --family_guidance \
      --probe_mode both --how_mode $MODE \
      --safety_scale $SS --attn_threshold $TXT --img_attn_threshold $IMG \
      --cas_threshold 0.6 --n_img_tokens 4 \
      --target_concepts "${TC[@]}" --anchor_concepts "${AC[@]}" --target_words "${TW[@]}" \
      </dev/null > "$LOGD/g${G}_${M}.log" 2>&1 &
    ;;
  *)
    echo "Unknown method: $M"; exit 1
    ;;
esac
echo "started SD14 sexual_q16top60 method=$M on g$G pid=$!"
