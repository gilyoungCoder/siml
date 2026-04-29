#!/bin/bash
set -uo pipefail
G="$1"; CAT="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGD=$REPO/CAS_SpatialCFG/exemplars/i2p_v1_flux1/_logs
mkdir -p "$LOGD"
setsid env CUDA_VISIBLE_DEVICES=$G "$PY" "$REPO/CAS_SpatialCFG/build_i2p_packs_flux1.py" --cats "$CAT" \
  </dev/null > "$LOGD/g${G}_${CAT}.log" 2>&1 &
echo "started FLUX1 pack build: g$G cat=$CAT pid=$!"
