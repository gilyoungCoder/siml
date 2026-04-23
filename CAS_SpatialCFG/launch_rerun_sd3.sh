#!/bin/bash
# Run on siml06 only — uses g4, g5
set -uo pipefail
HOST=$(hostname | tr '[:upper:]' '[:lower:]')
[ "$HOST" != "siml06" ] && { echo "ERROR: run only on siml06 (got $HOST)"; exit 1; }

REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LAUNCHER=$REPO/CAS_SpatialCFG/rerun_launcher.py
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0424_rerun_sd3/_gen_logs
mkdir -p "$LOGD"

GPUS=(4 5)
for i in 0 1; do
  G=${GPUS[$i]}
  setsid env CUDA_VISIBLE_DEVICES=$G "$PY" "$LAUNCHER" sd3 "$i" \
    </dev/null >"$LOGD/g${G}_w${i}.log" 2>&1 &
done
sleep 3
echo "[siml06 SD3] launched 2 workers (g4,g5), ps count:"
ps -ef | grep rerun_launcher | grep -v grep | wc -l
