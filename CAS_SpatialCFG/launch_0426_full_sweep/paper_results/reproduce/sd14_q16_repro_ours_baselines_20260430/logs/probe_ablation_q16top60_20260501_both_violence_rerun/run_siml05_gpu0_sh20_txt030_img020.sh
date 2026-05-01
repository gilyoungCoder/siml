#!/usr/bin/env bash
set -euo pipefail
GPU=0
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
CFG=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/configs/probe_ablation_q16top60_20260501_both_violence_rerun/sh20_tau04_txt030_img020.json
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/outputs/probe_ablation_q16top60_20260501_both_violence_rerun/sh20_tau04_txt030_img020
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_rerun
name=sh20_tau04_txt030_img020
rm -f "$OUT/results_qwen3_vl_violence_v5.txt"
echo "[$(date)] START siml-05 GPU0 $name" | tee -a "$LOGDIR/worker_siml05_gpu0.log"
CUDA_VISIBLE_DEVICES=$GPU REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$CFG" > "$LOGDIR/${name}_siml05_gpu0_gen.log" 2>&1
n=$(find "$OUT" -maxdepth 1 -name "*.png" | wc -l)
echo "[$(date)] GEN_DONE $name n=$n" | tee -a "$LOGDIR/worker_siml05_gpu0.log"
CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$OUT" violence qwen > "$LOGDIR/${name}_siml05_gpu0_eval.log" 2>&1
cat "$OUT/results_qwen3_vl_violence_v5.txt" | tee -a "$LOGDIR/worker_siml05_gpu0.log"
echo "[$(date)] DONE siml-05 GPU0 $name" | tee -a "$LOGDIR/worker_siml05_gpu0.log"
