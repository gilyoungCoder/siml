#!/bin/bash
# Launch SD3 SAFREE on i2p_q16_top60 single concept.
# usage: ./launch_sd3_safree_i2p.sh <gpu> <concept>
set -uo pipefail
G="$1"; C="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${C}_q16_top60.txt
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_safree_sd3/_logs
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_safree_sd3/$C
mkdir -p "$OUT" "$LOGD"

# concept name mapping (SD3 SAFREE uses 'selfharm' not 'self-harm')
case "$C" in
  self-harm) SF=selfharm ;;
  illegal_activity) SF=illegal ;;
  *) SF="$C" ;;
esac

setsid env CUDA_VISIBLE_DEVICES=$G $PY $REPO/scripts/sd3/generate_sd3_safree.py \
  --prompts "$PROMPTS" --outdir "$OUT" --concept $SF --seed 42 \
  </dev/null > "$LOGD/g${G}_${C}.log" 2>&1 &
echo "started SD3 SAFREE $C (sf=$SF) on g$G pid=$!"
