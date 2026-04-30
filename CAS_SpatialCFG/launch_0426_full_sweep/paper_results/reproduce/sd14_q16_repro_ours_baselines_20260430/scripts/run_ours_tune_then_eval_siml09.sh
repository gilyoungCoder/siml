#!/bin/bash
set -uo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
JOB=$ROOT/joblists/siml09_g0_ours_tune_all_missing_rerun.tsv
echo "[PIPE] generation start $(date)"
"$ROOT/scripts/worker.sh" "$GPU" "$JOB"
GEN_RC=$?
echo "[PIPE] generation done rc=$GEN_RC $(date)"
echo "[PIPE] eval start $(date)"
"$ROOT/scripts/eval_ours_tune_after_generation.sh" "$GPU"
EVAL_RC=$?
echo "[PIPE] eval done rc=$EVAL_RC $(date)"
exit $EVAL_RC
