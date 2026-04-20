#!/bin/bash
# NudeNet batch eval — scan all outdirs, eval if not done.
# Loops every 5 min for newly completed outdirs.
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

LOOP=0
while true; do
  LOOP=$((LOOP+1))
  echo "[$(date)] [nudenet loop #$LOOP]"
  PROCESSED=0
  for ROOT in launch_0420 launch_0420_i2p launch_0420_i2p_fullhard launch_0420_nudity; do
    while IFS= read -r outdir; do
      # Only nudity-related: rab, mja_sexual, sexual (i2p), unlearndiff, p4dn, mma, family_rab, v27_rab
      base=$(basename "$outdir")
      parent=$(basename "$(dirname "$outdir")")
      if [[ "$parent" != *"sexual"* && "$parent" != *"rab"* && "$parent" != *"unlearn"* && "$parent" != *"p4dn"* && "$parent" != *"mma"* && "$base" != *"sexual"* && "$base" != *"rab"* && "$base" != *"unlearn"* && "$base" != *"p4dn"* && "$base" != *"mma"* ]]; then continue; fi
      n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
      [ "$n_imgs" -lt 50 ] && continue  # too small
      out_json="$outdir/nudenet_results.json"
      if [ -f "$out_json" ]; then
        # check count match
        existing=$($PY -c "import json; d=json.load(open('$out_json')); print(len(d.get('per_image', d)))" 2>/dev/null || echo 0)
        [ "$existing" -ge "$n_imgs" ] && continue
      fi
      echo "[GPU $GPU] nudenet $outdir ($n_imgs imgs)"
      CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/scripts/launch_0420_i2p/nudenet_eval.py "$outdir" \
        >> $LOGDIR/nudenet_g${GPU}.log 2>&1 || echo "  failed"
      PROCESSED=$((PROCESSED+1))
    done < <(find $REPO/CAS_SpatialCFG/outputs/$ROOT -mindepth 2 -maxdepth 4 -type d 2>/dev/null)
  done
  echo "[$(date)] [nudenet] processed=$PROCESSED"
  if [ "$PROCESSED" = "0" ] && [ "$LOOP" -gt 4 ]; then echo "[done]"; break; fi
  sleep 300
done
