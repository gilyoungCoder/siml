#!/bin/bash
# NFE full orchestrator: 8 workers, slot offsets per server.
# Args: $1=OFFSET (0 for siml-03, 8 for siml-04). NSLOTS=16 total.
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/scripts/nfe_full.py
PY_HARNESS=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

echo "[$(date)] NFE full orch OFFSET=$OFFSET NSLOTS=$NSLOTS"
for gpu in 0 1 2 3 4 5 6 7; do
  slot=$((OFFSET + gpu))
  $PY_HARNESS $SCRIPT $gpu $slot $NSLOTS &
done
wait
echo "[$(date)] NFE full orch done"
