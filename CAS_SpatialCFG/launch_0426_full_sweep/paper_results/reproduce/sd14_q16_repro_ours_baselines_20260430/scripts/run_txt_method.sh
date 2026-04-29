#!/bin/bash
set -uo pipefail
METHOD=$1   # baseline|safree
CONCEPT=$2
GPU=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning
PY=${PY_SAFGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${CONCEPT}_q16_top60.txt
OUT=$ROOT/outputs/${METHOD}/i2p_q16/${CONCEPT}
mkdir -p "$OUT"
EXPECTED=$(grep -cve '^\s*$' "$PROMPTS")
COUNT=$(find "$OUT" -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$COUNT" -ge "$EXPECTED" ]; then echo "[SKIP $METHOD/$CONCEPT] count=$COUNT expected=$EXPECTED"; exit 0; fi
case "$METHOD" in
  baseline)
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/code/generate_baseline.py" \
      --nsamples 1 --prompts "$PROMPTS" --outdir "$OUT" --steps 50 --cfg_scale 7.5 --seed 42
    ;;
  safree)
    case "$CONCEPT" in sexual) SF_CAT=nudity ;; *) SF_CAT="$CONCEPT" ;; esac
    cd "$REPO/SAFREE"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" gen_safree_single.py \
      --txt "$PROMPTS" --save-dir "$OUT" \
      --model_id CompVis/stable-diffusion-v1-4 --category "$SF_CAT" \
      --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
      --seed 42 --image_length 512 --device cuda:0 --erase-id std \
      --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
      --safree -svf -lra --linear_per_prompt_seed
    ;;
  *) echo "unknown txt method $METHOD"; exit 2 ;;
esac
