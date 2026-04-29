#!/bin/bash
# Re-run hate best config with n_img_tokens=4 (was 16) — isolate n_tok effect.
# Source baseline: paper_results/single/hate/args.json (68.33% with n_tok=16)
# Only change: n_img_tokens 16 -> 4
# Server: siml-05 GPU 6 ONLY (do not touch other GPUs)
set -uo pipefail
GPU=${1:-6}
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN_DIR=/mnt/home3/yhgil99/unlearning/SafeGen
OUTDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_hate_ntok4_check/i2p_hate_ntok4_sh28_at0.25_ia0.05
mkdir -p "$OUTDIR"
cd "$SAFEGEN_DIR"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
  --prompts "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/i2p_sweep60/hate_sweep.txt" \
  --outdir "$OUTDIR" \
  --family_guidance \
  --family_config "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt" \
  --probe_mode both --probe_fusion union \
  --how_mode hybrid \
  --cas_threshold 0.6 \
  --safety_scale 28.0 \
  --attn_threshold 0.25 \
  --img_attn_threshold 0.05 \
  --n_img_tokens 4 \
  --steps 50 --seed 42 --cfg_scale 7.5 \
  --target_concepts "white supremacy" "racial slur" "hateful gathering" \
  --anchor_concepts "respect" "harmony" "peaceful gathering" \
  --target_words white supremacy racial slur hateful gathering

echo "DONE hate_ntok4: $OUTDIR"
