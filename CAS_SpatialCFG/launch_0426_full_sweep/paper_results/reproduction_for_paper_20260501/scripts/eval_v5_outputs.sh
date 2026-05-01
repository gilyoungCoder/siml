#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY_VLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
VLM=${VLM_SCRIPT:-/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py}
GPU=${GPU:-0}
run_eval(){ local dir=$1 concept=$2; [ -d "$dir" ] || { echo "[MISS] $dir"; return 0; }; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$VLM" "$dir" "$concept" qwen; }
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/sexual/hybrid_best_tau05_cas0.5" nudity
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/violence/hybrid_best_img075_img0.225" violence
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/self-harm/hybrid_best_tau05_cas0.5" self_harm
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/shocking/hybrid_best_ss125_ss27.5" shocking
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/illegal_activity/hybrid_best_ss125_ss25.0" illegal
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/harassment/hybrid_best_ss125_ss31.25" harassment
run_eval "$REPRO_ROOT/outputs/ours_best/i2p_q16/hate/hybrid_best_img075_img0.0375" hate
