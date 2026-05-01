#!/usr/bin/env bash
set -euo pipefail
DATASET=$1
GPU=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
LOG=$ROOT/logs/repair_sgf_nudity_breakdown_20260501
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PYVLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
mkdir -p "$LOG"
# Do not collide with current probe ablation on same GPU.
while pgrep -af "run_probe_ablation_q16_worker.sh $GPU " >/dev/null; do
  echo "[$(date)] wait probe ablation gpu=$GPU dataset=$DATASET" | tee -a "$LOG/${DATASET}_gpu${GPU}.log"
  sleep 120
done
OUT=$ROOT/outputs/sgf/nudity/$DATASET
RES=$OUT/all/results_qwen3_vl_nudity_v5.txt
if [ ! -s "$RES" ]; then
  echo "[$(date)] regenerate SGF nudity $DATASET gpu=$GPU" | tee -a "$LOG/${DATASET}_gpu${GPU}.log"
  bash "$ROOT/scripts/run_official_nudity.sh" sgf "$DATASET" "$GPU" 2>&1 | tee -a "$LOG/${DATASET}_gpu${GPU}.log"
  echo "[$(date)] eval SGF nudity $DATASET gpu=$GPU" | tee -a "$LOG/${DATASET}_gpu${GPU}.log"
  CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$OUT/all" nudity qwen 2>&1 | tee -a "$LOG/${DATASET}_eval_gpu${GPU}.log"
else
  echo "[$(date)] existing $RES" | tee -a "$LOG/${DATASET}_gpu${GPU}.log"
fi
