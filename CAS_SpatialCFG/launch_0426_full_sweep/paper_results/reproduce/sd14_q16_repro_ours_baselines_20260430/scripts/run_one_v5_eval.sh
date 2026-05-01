#!/usr/bin/env bash
set -uo pipefail
REL=$1
CONCEPT=$2
GPU=$3
NAME=$4
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/outputs/phase_safree_multi_q16top60
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/phase_safree_multi_q16top60_v5_0501_parallel
mkdir -p "$LOGDIR"
DIR="$BASE/$REL"
RES="$DIR/results_qwen3_vl_${CONCEPT}_v5.txt"; [ "$CONCEPT" = nudity ] && RES="$DIR/results_qwen3_vl_nudity_v5.txt"
if [ -s "$RES" ]; then echo "[SKIP existing] $NAME $RES"; exit 0; fi
N=$(find "$DIR" -maxdepth 1 -name "*.png" | wc -l)
echo "[$(date)] START $NAME gpu=$GPU concept=$CONCEPT n=$N dir=$DIR"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "$V5" "$DIR" "$CONCEPT" qwen 2>&1 | tee "$LOGDIR/${NAME}_gpu${GPU}.log"
