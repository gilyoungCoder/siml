#!/bin/bash
# Orchestrator: launches 8 workers in parallel (1 per GPU).
# Args: $1=SLOT_OFFSET (0 for siml-03, 8 for siml-05) — total NSLOTS=16
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
TSV=$BASE/cells_selfharm_push.tsv
WORKER=$BASE/scripts/worker_selfharm_push.sh
LOGDIR=$BASE/logs

echo "[$(date)] orchestrator OFFSET=$OFFSET NSLOTS=$NSLOTS"
for gpu in 0 1 2 3 4 5 6 7; do
  slot=$((OFFSET + gpu))
  bash $WORKER $gpu $slot $NSLOTS $TSV &
done
wait
echo "[$(date)] orchestrator done"
