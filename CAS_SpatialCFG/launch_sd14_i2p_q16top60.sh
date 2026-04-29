#!/bin/bash
# Launch SD1.4 baseline OR SAFREE on i2p_q16_top60 single concept.
# usage: ./launch_sd14_i2p_q16top60.sh <gpu> <method:baseline|safree> <concept>
set -uo pipefail
G="$1"; M="$2"; C="$3"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${C}_q16_top60.txt
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_${M}_sd14/_logs
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_${M}_sd14/$C
mkdir -p "$OUT" "$LOGD"

case "$M" in
  baseline)
    setsid env CUDA_VISIBLE_DEVICES=$G $PY $REPO/CAS_SpatialCFG/generate_baseline.py \
      --nsamples 1 --prompts "$PROMPTS" --outdir "$OUT" --steps 50 \
      </dev/null > "$LOGD/g${G}_${C}.log" 2>&1 &
    ;;
  safree)
    # SF concept mapping
    case "$C" in
      sexual) SF_CAT=nudity ;;
      *)      SF_CAT="$C" ;;
    esac
    cd $REPO/SAFREE
    setsid env CUDA_VISIBLE_DEVICES=$G $PY gen_safree_single.py \
      --txt "$PROMPTS" --save-dir "$OUT" \
      --model_id CompVis/stable-diffusion-v1-4 --category $SF_CAT \
      --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
      --seed 42 --image_length 512 --device cuda:0 --erase-id std \
      --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
      --safree -svf -lra --linear_per_prompt_seed \
      </dev/null > "$LOGD/g${G}_${C}.log" 2>&1 &
    ;;
  *) echo "unknown method: $M"; exit 1 ;;
esac
echo "started SD14 $M $C on g$G pid=$!"
