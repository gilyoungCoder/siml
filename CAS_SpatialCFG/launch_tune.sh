#!/bin/bash
# usage: launch_tune.sh <gpu> <sd3|flux1> <cat> <ss> <tau> [start end]
set -uo pipefail
G="$1"; BB="$2"; CAT="$3"; SS="$4"; TAU="$5"; START="${6:-}"; END="${7:-}"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_tune_${BB}/_logs
mkdir -p "$LOGD"
TAG="g${G}_${CAT}_ss${SS}_cas${TAU}"
[ -n "$START" ] && TAG="${TAG}_${START}-${END}"
EXTRA=""
[ -n "$START" ] && EXTRA="$START $END"
setsid env CUDA_VISIBLE_DEVICES=$G "$PY" "$REPO/CAS_SpatialCFG/tune_launcher.py" "$BB" "$CAT" "$SS" "$TAU" $EXTRA \
  </dev/null > "$LOGD/${TAG}.log" 2>&1 &
echo "started tune $BB $CAT ss=$SS tau=$TAU shard=[$START,$END] on g$G pid=$!"
