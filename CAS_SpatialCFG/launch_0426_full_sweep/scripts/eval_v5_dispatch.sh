#!/bin/bash
# Phase 3: v5 Qwen3-VL eval dispatcher.
# Builds list of (output_dir, eval_concept) tuples from generated cells, then dispatches across 8 GPUs.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR
LIST=$BASE/eval_pending.txt
> $LIST

# Phase 1 hybrid + Phase 1B anchor: same cell-base-name → rubric mapping.
# Anchor cells use the same base concept rubric (just different mode).
declare -A CELL2CONCEPT=(
  [nudity_ud]=nudity   [nudity_rab]=nudity   [nudity_p4dn]=nudity
  [mja_sexual]=nudity  [mja_violent]=violence  [mja_illegal]=illegal  [mja_disturbing]=shocking
  [i2p_violence]=violence  [i2p_self-harm]=self_harm  [i2p_shocking]=shocking
  [i2p_illegal]=illegal  [i2p_harassment]=harassment  [i2p_hate]=hate
  # anchor variants
  [nudity_ud_anchor]=nudity   [nudity_rab_anchor]=nudity   [nudity_p4dn_anchor]=nudity
  [mja_sexual_anchor]=nudity  [mja_violent_anchor]=violence  [mja_illegal_anchor]=illegal  [mja_disturbing_anchor]=shocking
  [i2p_violence_anchor]=violence  [i2p_self-harm_anchor]=self_harm  [i2p_shocking_anchor]=shocking
  [i2p_illegal_anchor]=illegal  [i2p_harassment_anchor]=harassment  [i2p_hate_anchor]=hate
)

# Phase 1 hybrid + Phase 1B anchor enumeration
for phase_dir in $BASE/outputs/phase1_single $BASE/outputs/phase1b_anchor; do
  [ -d "$phase_dir" ] || continue
  for d in $phase_dir/*/; do
    [ -d "$d" ] || continue
    cell=$(basename $d)
    if [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ]; then continue; fi
    concept=${CELL2CONCEPT[$cell]:-}
    if [ -z "$concept" ]; then
      echo "[$(date)] WARN: unknown cell $cell, skipping" | tee -a $LOGDIR/eval_v5.log
      continue
    fi
    json=$d/categories_qwen3_vl_${concept}_v5.json
    if [ ! -f "$json" ]; then
      echo "${d%/}|$concept" >> $LIST
    fi
  done
done

# Phase 2 multi enumeration
declare -A EVAL2RUBRIC=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)
for d in $BASE/outputs/phase2_multi/*/; do
  [ -d "$d" ] || continue
  name=$(basename $d)
  if [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ]; then continue; fi
  eval_part=${name##*__eval_}
  rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
  json=$d/categories_qwen3_vl_${rubric}_v5.json
  if [ ! -f "$json" ]; then
    echo "${d%/}|$rubric" >> $LIST
  fi
done

N=$(wc -l < $LIST)
echo "[$(date)] eval pending: $N cells" | tee -a $LOGDIR/eval_v5.log

if [ "$N" -eq 0 ]; then
  echo "Nothing to evaluate. Done."
  exit 0
fi

# Dispatch round-robin: 8 GPUs (slot 0..7), 1 process per GPU.
NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_v5_g${slot}.log
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json=$D/categories_qwen3_vl_${C}_v5.json
        if [ -f "$json" ]; then
          echo "[$(date)] [eval g$slot] SKIP $D $C" >> $WLOG
        else
          echo "[$(date)] [eval g$slot] EVAL $D $C" >> $WLOG
          cd $REPO/vlm
          CUDA_VISIBLE_DEVICES=$slot $PY $EVAL "$D" "$C" qwen >> $WLOG 2>&1
          rc=$?
          if [ $rc -ne 0 ]; then
            echo "[$(date)] [eval g$slot] FAIL $D $C exit=$rc" >> $WLOG
          else
            echo "[$(date)] [eval g$slot] DONE $D $C" >> $WLOG
          fi
        fi
      fi
      i=$((i+1))
    done < $LIST
    echo "[$(date)] [eval g$slot] worker complete" >> $WLOG
  ) &
done
wait
echo "[$(date)] All eval workers complete" | tee -a $LOGDIR/eval_v5.log
