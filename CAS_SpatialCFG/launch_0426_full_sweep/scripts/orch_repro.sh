#!/bin/bash
# Reproducibility orchestrator: 8 workers on siml-04 GPUs.
set -uo pipefail
NSLOTS=8
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
RUNNER=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/scripts/repro_runner.py
echo "[$(date)] repro orch start NSLOTS=$NSLOTS"
for gpu in 0 1 2 3 4 5 6 7; do
  $PY $RUNNER $gpu $NSLOTS $gpu &
done
wait
echo "[$(date)] repro orch done"
