#!/bin/bash
# Run on siml01/siml02/siml05 — each launches 8 workers (24 jobs total).
set -uo pipefail
HOST=$(hostname | tr '[:upper:]' '[:lower:]')
case $HOST in
  siml01) START=0 ;;
  siml02) START=8 ;;
  siml05) START=16 ;;
  *) echo "ERROR: run only on siml01/02/05 (got $HOST)"; exit 1 ;;
esac
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LAUNCHER=$REPO/CAS_SpatialCFG/rerun_v3.py
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0424_v3/_gen_logs
mkdir -p "$LOGD"

for g in 0 1 2 3 4 5 6 7; do
  W=$((START + g))
  setsid env CUDA_VISIBLE_DEVICES=$g "$PY" "$LAUNCHER" "$W" \
    </dev/null >"$LOGD/${HOST}_g${g}_w${W}.log" 2>&1 &
done
sleep 3
echo "[$HOST] launched 8 workers (start=$START), ps count:"
ps -ef | awk '$1=="yhgil99" && /rerun_v3/' | wc -l
