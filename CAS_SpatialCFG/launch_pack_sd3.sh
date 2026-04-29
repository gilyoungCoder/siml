#!/bin/bash
# Launch SD3 pack build, one concept per GPU. usage: ./launch_pack_sd3.sh <gpu> <cat>
set -uo pipefail
HOST=$(hostname | tr '[:upper:]' '[:lower:]')
[ "$HOST" != "siml07" ] && { echo "ERROR: run on siml07 (got $HOST)"; exit 1; }
G="$1"; CAT="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGD=$REPO/CAS_SpatialCFG/exemplars/i2p_v1_sd3/_logs
mkdir -p "$LOGD"
setsid env CUDA_VISIBLE_DEVICES=$G "$PY" "$REPO/CAS_SpatialCFG/build_i2p_packs_sd3.py" --cats "$CAT" \
  </dev/null > "$LOGD/g${G}_${CAT}.log" 2>&1 &
echo "started SD3 pack build: g$G cat=$CAT pid=$!"
