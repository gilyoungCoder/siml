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
> $LIST  # empty

# Map cell name → eval concept rubric for v5 eval (concept arg of the eval script)
# Phase 1 cells:
declare -A CELL2CONCEPT=(
  [nudity_ud]=nudity   [nudity_rab]=nudity   [nudity_p4dn]=nudity
  [mja_sexual]=nudity  [mja_violent]=violence  [mja_illegal]=illegal  [mja_disturbing]=shocking
  [i2p_violence]=violence  [i2p_self-harm]=self_harm  [i2p_shocking]=shocking
  [i2p_illegal]=illegal  [i2p_harassment]=harassment  [i2p_hate]=hate
)

# Phase 1: enumerate single-concept dirs
for cell in "${!CELL2CONCEPT[@]}"; do
  dir=$BASE/outputs/phase1_single/$cell
  if [ -d "$dir" ] && [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    concept=${CELL2CONCEPT[$cell]}
    json=$dir/categories_qwen3_vl_${concept}_v5.json
    if [ ! -f "$json" ]; then
      echo "$dir|$concept" >> $LIST
    fi
  fi
done

# Phase 2: enumerate multi dirs and use eval_concept from cell name.
# Cell name pattern: <setup>__<config>__eval_<concept>
# Map cell eval_concept → v5 rubric concept (sexual→nudity, others 1:1, but illegal_activity→illegal, self-harm→self_harm)
declare -A EVAL2RUBRIC=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)
for dir in $BASE/outputs/phase2_multi/*/; do
  name=$(basename $dir)
  if [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -eq 0 ]; then continue; fi
  eval_part=${name##*__eval_}
  rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
  json=$dir/categories_qwen3_vl_${rubric}_v5.json
  if [ ! -f "$json" ]; then
    echo "${dir%/}|$rubric" >> $LIST
  fi
done

N=$(wc -l < $LIST)
echo "[$(date)] eval pending: $N cells" | tee -a $LOGDIR/eval_v5.log

if [ "$N" -eq 0 ]; then
  echo "Nothing to evaluate. Done."
  exit 0
fi

# Dispatch round-robin: 8 GPUs (slot 0..7), 1 process per GPU (Qwen3-VL is heavy ~17GB).
NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_v5_g${slot}.log
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json=$D/categories_qwen3_vl_${C}_v5.json
        if [ -f "$json" ]; then
          echo "[$(date)] [eval g$slot] SKIP (exists) $D $C" >> $WLOG
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
