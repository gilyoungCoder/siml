#!/bin/bash
# Orchestrator v3: 8 workers in parallel.
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
TSV=$BASE/cells_selfharm_v3.tsv
WORKER=$BASE/scripts/worker_selfharm_v3.sh

echo "[$(date)] v3 orchestrator OFFSET=$OFFSET NSLOTS=$NSLOTS"
for gpu in 0 1 2 3 4 5 6 7; do
  slot=$((OFFSET + gpu))
  bash $WORKER $gpu $slot $NSLOTS $TSV &
done
wait
echo "[$(date)] v3 orchestrator done"
