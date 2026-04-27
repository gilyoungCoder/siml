#!/bin/bash
# Eval phase_selfharm_push (32 cells × 60 imgs) on 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
LIST=$BASE/eval_selfharm_push_pending.txt
> $LIST

for d in $BASE/outputs/phase_selfharm_push/*/; do
  [ -d "$d" ] || continue
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -gt 0 ] || continue
  json="${d%/}/categories_qwen3_vl_self_harm_v5.json"
  [ -f "$json" ] || echo "${d%/}|self_harm" >> $LIST
done

N=$(wc -l < $LIST)
echo "[$(date)] eval_selfharm_push pending: $N cells" | tee -a $LOGDIR/eval_selfharm_push.log
[ "$N" -eq 0 ] && exit 0

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_selfharm_push_g${slot}.log
  > "$WLOG"
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json="$D/categories_qwen3_vl_${C}_v5.json"
        if [ -f "$json" ]; then
          echo "[$(date)] [g$slot] SKIP $D" >> "$WLOG"
        else
          echo "[$(date)] [g$slot] EVAL $D" >> "$WLOG"
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
echo "[$(date)] all workers done" | tee -a $LOGDIR/eval_selfharm_push.log
