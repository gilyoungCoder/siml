#!/bin/bash
# Reproduce paper Table 8/9 multi-concept hybrid result for: 1c_sexual
# Source: CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/1c_sexual/sexual/args.json
# Usage: bash run_1c_sexual.sh [GPU=0] [OUTDIR=./reproduce_out/1c_sexual]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/1c_sexual}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen

mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family_multi \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/sexual_sweep.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt" \
  --probe_mode both \
  --how_mode hybrid \
  --probe_fusion union \
  --cas_threshold 0.6 \
  --safety_scale 15.0 \
  --attn_threshold 0.1 \
  --img_attn_threshold 0.3 \
  --steps 50 \
  --seed 42 \
  --cfg_scale 7.5 \
  --target_concepts 'nudity' 'nude_person' 'naked_body' 

echo "DONE 1c_sexual: $OUTDIR"
