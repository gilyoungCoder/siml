#!/bin/bash
# Reproduce nudity benchmark hybrid result: Ring-A-Bell (RAB)
# Usage: bash run_nudity_rab.sh [GPU=0] [OUTDIR=./reproduce_out/nudity_rab]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/nudity_rab}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen
mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt" \
  --probe_mode both --how_mode hybrid --probe_fusion union \
  --cas_threshold 0.6 --safety_scale 20.0 \
  --attn_threshold 0.1 --img_attn_threshold 0.4 --n_img_tokens 4 \
  --steps 50 --seed 42 --cfg_scale 7.5 \
  --target_concepts nudity "nude person" "naked body" \
  --anchor_concepts "clothed person" "person wearing clothes"
echo "DONE nudity_rab: $OUTDIR"
