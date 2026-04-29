#!/bin/bash
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt
PROMPTS=$REPO/CAS_SpatialCFG/sanity_0429/violence5.txt
OUT=$REPO/CAS_SpatialCFG/sanity_0429/sd3_violence_hybrid

TC=("bloody wounds" "weapon threat" "physical combat" "dramatic distress")
AC=("clean skin" "man with tool" "friendly handshake" "smiling face")

CUDA_VISIBLE_DEVICES=0 $PY $REPO/scripts/sd3/generate_sd3_safegen.py \
  --prompts "$PROMPTS" --outdir "$OUT" \
  --family_config "$PACK" --family_guidance \
  --probe_mode both --how_mode hybrid \
  --safety_scale 20.0 --attn_threshold 0.15 --img_attn_threshold 0.10 \
  --cas_threshold 0.5 --n_img_tokens 4 \
  --target_concepts "${TC[@]}" --anchor_concepts "${AC[@]}"
