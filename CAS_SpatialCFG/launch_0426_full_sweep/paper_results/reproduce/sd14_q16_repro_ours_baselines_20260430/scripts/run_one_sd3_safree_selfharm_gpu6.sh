#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
WRAP=$ROOT/scripts/run_sd3_with_sfgd_torch_safree_libs.py
GEN=$ROOT/scripts/generate_sd3_prompt_repellency.py
c=self-harm
out=$ROOT/outputs/crossbackbone_0501/sd3/safree/i2p_q16/$c
prompt=$ROOT/prompts/i2p_q16_csv/${c}_q16_top60.csv
rm -rf "$out"; mkdir -p "$out"
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 "$PY" "$WRAP" "$GEN" --method safree --prompts "$prompt" --outdir "$out" --steps 28 --guidance_scale 7.0 --height 1024 --width 1024 --device cuda:0
