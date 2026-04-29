#!/bin/bash
# Eval phase_hate_v2 (192 cells) + phase_repro/7c_all on 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
LIST=$BASE/eval_hatev2_7c_pending.txt
> $LIST

# hate v2 cells
for d in $BASE/outputs/phase_hate_v2/*/; do
  [ -d "$d" ] || continue
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
  json="${d%/}/categories_qwen3_vl_hate_v5.json"
  [ -f "$json" ] || echo "${d%/}|hate" >> $LIST
done

# 7c_all repro
d="$BASE/outputs/phase_repro/7c_all"
[ -d "$d" ] && [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] && \
  [ ! -f "$d/categories_qwen3_vl_nudity_v5.json" ] && echo "$d|nudity" >> $LIST

N=$(wc -l < $LIST)
echo "[$(date)] eval pending: $N cells (hate v2 + 7c)" | tee -a $LOGDIR/eval_hatev2_7c.log
[ "$N" -eq 0 ] && exit 0

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_hatev2_7c_g${slot}.log
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
echo "[$(date)] all workers done" | tee -a $LOGDIR/eval_hatev2_7c.log
