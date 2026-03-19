#!/usr/bin/env bash
set -Eeuo pipefail

# (선택) GPU 지정
export CUDA_VISIBLE_DEVICES=7
# ../SoftDelete+CG/prompts/country_nude_body.txt
python gen_safree_simple.py \
  --txt ../SoftDelete+CG/prompts/violence_50.txt \
  --model_id CompVis/stable-diffusion-v1-4 \
  --outdir ../SoftDelete+CG/reference/SAFREE/violence_last \
  --num_images 1 --steps 50 --guidance 7.5 --seed 1234 \
  --safree --lra --svf --sf_alpha 0.01 --re_attn_t=-1,4 --up_t 10 \
  --use_default_negative

