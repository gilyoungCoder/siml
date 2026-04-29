#!/bin/bash
# Reproduce paper Table 1 single-concept hybrid result for: self-harm
# Source: CAS_SpatialCFG/launch_0426_full_sweep/paper_results/single/self-harm/args.json
# Usage: bash run_self-harm.sh [GPU=0] [OUTDIR=./reproduce_out/self-harm]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/self-harm}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen

mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/self-harm_sweep.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt" \
  --probe_mode both \
  --how_mode hybrid \
  --probe_fusion union \
  --cas_threshold 0.6 \
  --safety_scale 7.0 \
  --attn_threshold 0.1 \
  --img_attn_threshold 0.1 \
  --n_img_tokens 16 \
  --steps 50 \
  --seed 42 \
  --cfg_scale 7.5 \
  --target_concepts 'crying despair' 'hanging noose' 'self mutilation' 

echo "DONE self-harm: $OUTDIR"
