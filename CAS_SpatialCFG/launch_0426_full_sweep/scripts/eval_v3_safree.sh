#!/bin/bash
# Combined eval: phase_selfharm_v3 + phase_safree_v2
# Args: $1=PHASE (v3 or safree_v2)
set -uo pipefail
PHASE=${1:?need PHASE arg}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
LIST=$BASE/eval_${PHASE}_pending.txt
> $LIST

declare -A EVAL2RUBRIC=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)

if [ "$PHASE" = "v3" ]; then
  PHASE_DIR=phase_selfharm_v3
  for d in $BASE/outputs/$PHASE_DIR/*/; do
    [ -d "$d" ] || continue
    [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
    json="${d%/}/categories_qwen3_vl_self_harm_v5.json"
    [ -f "$json" ] || echo "${d%/}|self_harm" >> $LIST
  done
elif [ "$PHASE" = "safree_v2" ]; then
  PHASE_DIR=phase_safree_v2
  for d in $BASE/outputs/$PHASE_DIR/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    # Both i2p_<concept> and Nc_<setup>__eval_<concept>
    if [[ "$name" == *"__eval_"* ]]; then
      eval_part=${name##*__eval_}
    else
      eval_part=${name##i2p_}
    fi
    rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
    # PNG location: SAFREE writes to outdir/generated/ then was supposed to move; check both
    img_dir="$d"
    if [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -lt 60 ] && [ -d "$d/generated" ]; then
      img_dir="$d/generated"
    fi
    [ "$(ls "$img_dir"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
    json="${img_dir%/}/categories_qwen3_vl_${rubric}_v5.json"
    [ -f "$json" ] || echo "${img_dir%/}|${rubric}" >> $LIST
  done
fi

N=$(wc -l < $LIST)
echo "[$(date)] eval $PHASE pending: $N cells" | tee -a $LOGDIR/eval_${PHASE}.log
[ "$N" -eq 0 ] && exit 0

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_${PHASE}_g${slot}.log
  > "$WLOG"
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json="$D/categories_qwen3_vl_${C}_v5.json"
        if [ -f "$json" ]; then
          echo "[$(date)] [g$slot] SKIP $D" >> "$WLOG"
        else
          echo "[$(date)] [g$slot] EVAL $D $C" >> "$WLOG"
          cd $REPO/vlm
          CUDA_VISIBLE_DEVICES=$slot $PY $EVAL "$D" "$C" qwen >> "$WLOG" 2>&1
          rc=$?
          if [ $rc -ne 0 ]; then
            echo "[$(date)] [g$slot] FAIL $D rc=$rc" >> "$WLOG"
          else
            echo "[$(date)] [g$slot] DONE $D" >> "$WLOG"
          fi
        fi
      fi
      i=$((i+1))
    done < $LIST
  ) &
done
wait
echo "[$(date)] all workers done" | tee -a $LOGDIR/eval_${PHASE}.log
