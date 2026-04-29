#!/bin/bash
set -uo pipefail
GPU=$1
JOBLIST=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd3_flux1_q16_7concept_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
echo "[variant-worker g$GPU] START $(date) list=$(basename "$JOBLIST")"
while IFS=$'\t' read -r BB C SS TAU TAG; do
  [[ -z "${BB:-}" || "$BB" =~ ^# ]] && continue
  echo "[variant-worker g$GPU] JOB $BB $C ss=$SS tau=$TAU tag=$TAG $(date)"
  "$PY" "$ROOT/scripts/run_variant.py" --backbone "$BB" --concept "$C" --ss "$SS" --tau "$TAU" --tag "$TAG" --gpu "$GPU"
  rc=$?
  echo "[variant-worker g$GPU] DONE $BB $C rc=$rc $(date)"
  if [ $rc -ne 0 ]; then echo "[variant-worker g$GPU] continue rc=$rc"; fi
done < "$JOBLIST"
echo "[variant-worker g$GPU] ALL_DONE $(date)"
