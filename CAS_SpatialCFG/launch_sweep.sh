#!/bin/bash
# Launch one sweep worker.  usage: ./launch_sweep.sh <gpu> <backbone> <widx>
set -uo pipefail
G="$1"; BB="$2"; WIDX="$3"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_${BB}/_logs
mkdir -p "$LOGD"
setsid env CUDA_VISIBLE_DEVICES=$G "$PY" "$REPO/CAS_SpatialCFG/i2p_sweep_launcher.py" "$BB" "$WIDX" \
  </dev/null > "$LOGD/g${G}_w${WIDX}.log" 2>&1 &
echo "started $BB widx=$WIDX on g$G pid=$!"
