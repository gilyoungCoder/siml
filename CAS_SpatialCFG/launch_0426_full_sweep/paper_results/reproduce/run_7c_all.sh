#!/bin/bash
# Reproduce paper Table 8/9 multi-concept hybrid result for: 7c_all
# Source: CAS_SpatialCFG/launch_0426_full_sweep/paper_results/multi/7c_all/sexual/args.json
# Usage: bash run_7c_all.sh [GPU=0] [OUTDIR=./reproduce_out/7c_all]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/7c_all}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen

mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family_multi \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/sexual_sweep.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/self-harm/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/shocking/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/illegal_activity/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/harassment/clip_grouped.pt" "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt" \
  --probe_mode both \
  --how_mode hybrid \
  --probe_fusion union \
  --cas_threshold 0.6 0.6 0.6 0.6 0.6 0.6 0.6 \
  --safety_scale 19.5 19.5 28.6 28.6 26.0 26.0 28.6 \
  --attn_threshold 0.1 0.1 0.1 0.15 0.1 0.15 0.25 \
  --img_attn_threshold 0.3 0.3 0.4 0.1 0.5 0.1 0.1 \
  --steps 50 \
  --seed 42 \
  --cfg_scale 7.5 \
  --target_concepts 'nudity' 'nude_person' 'naked_body' 

echo "DONE 7c_all: $OUTDIR"
