#!/bin/bash
# Eval NFE v2 (132 cells) + image saturation (3 cells) on 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
LIST=$BASE/eval_nfe_img_pending.txt
> $LIST

declare -A C2R=(
  [violence]=violence  [shocking]=shocking
  [self-harm]=self_harm  [sexual]=nudity
)

# phase_nfe_full: cells like METHOD_CONCEPT_stepsN
for d in $BASE/outputs/phase_nfe_full/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
  # Extract concept from name: METHOD_CONCEPT_stepsN where CONCEPT may have hyphen (self-harm)
  rest=${name#*_}                    # drop method prefix
  concept_part=${rest%_steps*}       # drop _stepsN suffix
  rubric=${C2R[$concept_part]:-$concept_part}
  json="${d%/}/categories_qwen3_vl_${rubric}_v5.json"
  [ -f "$json" ] || echo "${d%/}|${rubric}" >> $LIST
done

# phase_img_saturation: sexual_K{N}
for d in $BASE/outputs/phase_img_saturation/*/; do
  [ -d "$d" ] || continue
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
  json="${d%/}/categories_qwen3_vl_nudity_v5.json"
  [ -f "$json" ] || echo "${d%/}|nudity" >> $LIST
done

N=$(wc -l < $LIST)
echo "[$(date)] eval pending: $N cells (NFE v2 + img saturation)" | tee -a $LOGDIR/eval_nfe_img.log
[ "$N" -eq 0 ] && exit 0

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_nfe_img_g${slot}.log
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
          [ $rc -ne 0 ] && echo "[$(date)] [g$slot] FAIL $D rc=$rc" >> "$WLOG" || \
                           echo "[$(date)] [g$slot] DONE $D" >> "$WLOG"
        fi
      fi
      i=$((i+1))
    done < $LIST
  ) &
done
wait
echo "[$(date)] eval done" | tee -a $LOGDIR/eval_nfe_img.log
