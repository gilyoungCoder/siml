#!/bin/bash
# Eval phase_repro + phase_nfe_ablation on 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=$REPO/.conda/envs/vlm/bin/python3.10
[ -x /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 ] && PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
LIST=$BASE/eval_repro_nfe_pending.txt
> $LIST

# Concept → rubric
declare -A C2R=(
  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal]=illegal  [harassment]=harassment  [hate]=hate
  [nudity_p4dn]=nudity  [nudity_rab]=nudity  [nudity_ud]=nudity  [nudity_mma]=nudity
)
declare -A E2R=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)

# phase_repro
for d in $BASE/outputs/phase_repro/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  png_count=$(ls "$d"/*.png 2>/dev/null | wc -l)
  [ "$png_count" -ge 60 ] || continue
  # determine rubric
  if [[ "$name" == 1c_* ]]; then rubric=nudity
  elif [[ "$name" == 2c_* ]]; then rubric=nudity   # eval sexual cell only for repro
  elif [[ "$name" == 3c_* ]]; then rubric=nudity
  elif [[ "$name" == 7c_* ]]; then rubric=nudity
  else rubric=${C2R[$name]:-}
  fi
  [ -n "$rubric" ] || { echo "WARN unknown $name"; continue; }
  json="${d%/}/categories_qwen3_vl_${rubric}_v5.json"
  [ -f "$json" ] || echo "${d%/}|${rubric}" >> $LIST
done

# phase_nfe_ablation
for d in $BASE/outputs/phase_nfe_ablation/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -ge 60 ] || continue
  # name = <concept>_steps<N> e.g. self-harm_steps5
  base=${name%_steps*}
  rubric=${C2R[$base]:-$base}
  json="${d%/}/categories_qwen3_vl_${rubric}_v5.json"
  [ -f "$json" ] || echo "${d%/}|${rubric}" >> $LIST
done

N=$(wc -l < $LIST)
echo "[$(date)] eval pending: $N cells (repro+nfe)" | tee -a $LOGDIR/eval_repro_nfe.log
[ "$N" -eq 0 ] && exit 0

NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_repro_nfe_g${slot}.log
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
echo "[$(date)] all workers done" | tee -a $LOGDIR/eval_repro_nfe.log
