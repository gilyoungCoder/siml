#!/usr/bin/env bash
set -euo pipefail
GPU=$1; START=$2; END=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SAFEGEN=$ROOT/code/SafeGen
OUT=$ROOT/outputs/ours/coco_family_best/chunks/${START}_${END}
mkdir -p "$OUT"
cd "$SAFEGEN"
echo "[RUN family COCO] gpu=$GPU start=$START end=$END out=$OUT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" -m safegen.generate_family \
  --ckpt CompVis/stable-diffusion-v1-4 \
  --prompts "$CAS/prompts/coco_250.txt" \
  --outdir "$OUT" \
  --nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 \
  --start_idx "$START" --end_idx "$END" \
  --cas_threshold 0.6 --probe_mode both \
  --family_config "$CAS/exemplars/concepts_v2/sexual/clip_grouped.pt" \
  --attn_threshold 0.1 --img_attn_threshold 0.3 --attn_sigmoid_alpha 10.0 --blur_sigma 1.0 \
  --probe_fusion union --n_img_tokens 4 \
  --how_mode hybrid --safety_scale 20.0 \
  --attn_resolutions 16 32 \
  --target_words nudity nude person naked body \
  --family_guidance \
  --target_concepts nudity "nude person" "naked body" \
  --anchor_concepts "clothed person" "person wearing clothes"
