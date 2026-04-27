#!/bin/bash
# Eval remaining cells in phase2_multi only.
# Skip GPU 1 (occupied by own sdxl_lightning eval).
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR
LIST=$BASE/eval_phase2multi_pending.txt
> $LIST

declare -A EVAL2RUBRIC=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)

for d in $BASE/outputs/phase2_multi/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -gt 0 ] || continue
  eval_part=${name##*__eval_}
  rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
  json="${d%/}/categories_qwen3_vl_${rubric}_v5.json"
  if [ ! -f "$json" ]; then
    echo "${d%/}|${rubric}" >> $LIST
  fi
done

N=$(wc -l < $LIST)
echo "[$(date)] phase2_multi eval pending: $N cells" | tee -a $LOGDIR/eval_phase2multi.log
[ "$N" -eq 0 ] && exit 0

# Use 7 GPUs (skip GPU 1 due to existing job).
GPUS=(0 2 3 4 5 6 7)
NSLOTS=${#GPUS[@]}
for slot in "${!GPUS[@]}"; do
  G=${GPUS[$slot]}
  WLOG=$LOGDIR/eval_phase2multi_g${G}.log
  > "$WLOG"
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json="$D/categories_qwen3_vl_${C}_v5.json"
        if [ -f "$json" ]; then
          echo "[$(date)] [g$G] SKIP $D $C" >> "$WLOG"
        else
          echo "[$(date)] [g$G] EVAL $D $C" >> "$WLOG"
          cd $REPO/vlm
          CUDA_VISIBLE_DEVICES=$G $PY $EVAL "$D" "$C" qwen >> "$WLOG" 2>&1
          rc=$?
          if [ $rc -ne 0 ]; then
            echo "[$(date)] [g$G] FAIL $D $C exit=$rc" >> "$WLOG"
          else
            echo "[$(date)] [g$G] DONE $D $C" >> "$WLOG"
          fi
        fi
      fi
      i=$((i+1))
    done < $LIST
    echo "[$(date)] [g$G] worker complete" >> "$WLOG"
  ) &
done
wait
echo "[$(date)] all workers complete" | tee -a $LOGDIR/eval_phase2multi.log
