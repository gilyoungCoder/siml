#!/bin/bash
# COCO FID orchestrator: run 3 methods sequentially, each across 8 GPUs.
# Args: $1=OFFSET (0 for siml-03, 8 for siml-04). NSLOTS=16 across both servers.
set -uo pipefail
OFFSET=${1:-0}
NSLOTS=16
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/CAS_SpatialCFG/launch_0426_full_sweep/scripts/coco_fid_runner.py
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

for METHOD in ebsg safree baseline; do
  echo "[$(date)] starting $METHOD OFFSET=$OFFSET"
  for gpu in 0 1 2 3 4 5 6 7; do
    slot=$((OFFSET + gpu))
    $PY $SCRIPT $gpu $slot $NSLOTS $METHOD &
  done
  wait
  echo "[$(date)] $METHOD wave done"
done
echo "[$(date)] all 3 methods done"
