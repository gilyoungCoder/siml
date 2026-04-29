#!/bin/bash
# Reproduce nudity benchmark hybrid result: MMA. Optional [START_IDX] [END_IDX] support allows split runs.
# Usage: bash run_nudity_mma.sh [GPU=0] [OUTDIR=./reproduce_out/nudity_mma] [START_IDX=0] [END_IDX=-1]
set -uo pipefail
GPU=${1:-0}
OUTDIR=${2:-./reproduce_out/nudity_mma}
START_IDX=${3:-0}
END_IDX=${4:--1}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen
mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mma.txt" \
  --outdir "$OUTDIR" \
  --start_idx "$START_IDX" --end_idx "$END_IDX" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt" \
  --probe_mode both --how_mode hybrid --probe_fusion union \
  --cas_threshold 0.6 --safety_scale 20.0 \
  --attn_threshold 0.1 --img_attn_threshold 0.3 --n_img_tokens 4 \
  --steps 50 --seed 42 --cfg_scale 7.5 \
  --target_concepts nudity "nude person" "naked body" \
echo "DONE nudity_mma: $OUTDIR [$START_IDX,$END_IDX)"
