#!/bin/bash
# NFE full v2 orchestrator: 8 workers, slot offset per server.
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/scripts/nfe_full_v2.py
PY_HARNESS=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

echo "[$(date)] NFE v2 orch OFFSET=$OFFSET NSLOTS=$NSLOTS"
for gpu in 0 1 2 3 4 5 6 7; do
  slot=$((OFFSET + gpu))
  $PY_HARNESS $SCRIPT $gpu $slot $NSLOTS &
done
wait
echo "[$(date)] NFE v2 orch done"
