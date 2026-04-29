#!/bin/bash
set -uo pipefail
GPU=$1
WIDX=$2
CONCEPT=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd3_flux1_q16_7concept_20260430
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LAUNCH=$REPO/CAS_SpatialCFG/i2p_sweep_launcher.py
echo "[g$GPU/$CONCEPT] START $(date)"
for BB in sd3 flux1; do
  echo "[g$GPU/$CONCEPT] RUN $BB $(date)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$LAUNCH" "$BB" "$WIDX"
  rc=$?
  echo "[g$GPU/$CONCEPT] DONE $BB rc=$rc $(date)"
  if [ $rc -ne 0 ]; then echo "[g$GPU/$CONCEPT] continue after rc=$rc"; fi
done
echo "[g$GPU/$CONCEPT] ALL_DONE $(date)"
