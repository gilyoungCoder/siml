#!/bin/bash
# Reproduce paper Table 1 single-concept hybrid result for: hate
# Source: CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/hate/args.json
# Usage: bash run_hate.sh [GPU=0] [OUTDIR=./reproduce_out/hate]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/hate}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen

mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/hate_sweep.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt" \
  --probe_mode both \
  --how_mode hybrid \
  --probe_fusion union \
  --cas_threshold 0.6 \
  --safety_scale 28.0 \
  --attn_threshold 0.25 \
  --img_attn_threshold 0.05 \
  --n_img_tokens 16 \
  --steps 50 \
  --seed 42 \
  --cfg_scale 7.5 \
  --target_concepts 'white supremacy' 'racial slur' 'hateful gathering' \
  --anchor_concepts 'respect' 'harmony' 'peaceful gathering'

echo "DONE hate: $OUTDIR"
