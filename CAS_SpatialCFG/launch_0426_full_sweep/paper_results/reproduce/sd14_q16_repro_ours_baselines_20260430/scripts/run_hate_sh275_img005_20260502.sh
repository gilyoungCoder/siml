#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CFG=$ROOT/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round/args.json
OUT=$ROOT/outputs/ours_round_sh_sanity_20260502/i2p_q16/hate/hybrid_sh27.5_cas0.6_txt0.25_img0.05_round
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PYV=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
GPU=${GPU:-4}
echo "[$(date '+%F %T')] START hate gen gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 "$ROOT/scripts/run_ours_from_config.py" --config "$CFG" --gpu "$GPU" --expected 60
echo "[$(date '+%F %T')] START hate eval gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU "$PYV" "$V5" "$OUT" hate qwen | tee "$OUT/results_qwen3_vl_hate_v5.txt"
echo "[$(date '+%F %T')] DONE hate"
