#!/bin/bash
# Hate v2 orchestrator: 8 workers parallel.
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
TSV=$BASE/cells_hate_v2.tsv
WORKER=$BASE/scripts/worker_hate_v2.sh

echo "[$(date)] hate v2 orchestrator OFFSET=$OFFSET"
for gpu in 0 1 2 3 4 5 6 7; do
  slot=$((OFFSET + gpu))
  bash $WORKER $gpu $slot $NSLOTS $TSV &
done
wait
echo "[$(date)] hate v2 orch done"
