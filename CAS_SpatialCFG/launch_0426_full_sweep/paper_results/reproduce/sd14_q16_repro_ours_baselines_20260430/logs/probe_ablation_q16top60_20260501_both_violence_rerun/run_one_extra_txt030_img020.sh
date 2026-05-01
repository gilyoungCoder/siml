#!/usr/bin/env bash
set -euo pipefail
GPU=3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
CFG=$ROOT/configs/probe_ablation_q16top60_20260501_both_violence_rerun/sh195_tau04_txt030_img020.json
OUT=$ROOT/outputs/probe_ablation_q16top60_20260501_both_violence_rerun/sh195_tau04_txt030_img020
LOGDIR=$ROOT/logs/probe_ablation_q16top60_20260501_both_violence_rerun
name=sh195_tau04_txt030_img020
while pgrep -af "safegen.generate_family.*probe_ablation_q16top60_20260501_both_violence_rerun" >/dev/null; do sleep 60; done
echo "[$(date)] START EXTRA $name on GPU=$GPU" | tee -a "$LOGDIR/worker_gpu3.log"
CUDA_VISIBLE_DEVICES=$GPU REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$CFG" > "$LOGDIR/${name}_gen.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$OUT" violence qwen > "$LOGDIR/${name}_eval.log" 2>&1
cat "$OUT/results_qwen3_vl_violence_v5.txt" | tee -a "$LOGDIR/worker_gpu3.log"
echo "[$(date)] DONE EXTRA $name" | tee -a "$LOGDIR/worker_gpu3.log"
