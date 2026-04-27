#!/bin/bash
# Phase 3 v2: v5 Qwen3-VL eval dispatcher across the 5 new phase dirs.
# Phases handled: phase_paper_best, phase_tune, phase_selfharm, phase_multi_v3, phase_safree.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOGDIR=$BASE/logs
mkdir -p $LOGDIR
LIST=$BASE/eval_pending_v2.txt
> $LIST

# ---- Mapping: base cell name → rubric concept (v5 evaluator)
declare -A CELL2CONCEPT=(
  [nudity_ud]=nudity   [nudity_rab]=nudity   [nudity_p4dn]=nudity   [nudity_mma]=nudity
  [mja_sexual]=nudity  [mja_violent]=violence  [mja_illegal]=illegal  [mja_disturbing]=shocking
  [i2p_violence]=violence  [i2p_self-harm]=self_harm  [i2p_shocking]=shocking
  [i2p_illegal]=illegal  [i2p_harassment]=harassment  [i2p_hate]=hate
)

# Eval-suffix → rubric (used by phase_multi_v3 + phase_safree)
declare -A EVAL2RUBRIC=(
  [sexual]=nudity  [violence]=violence  [self-harm]=self_harm  [shocking]=shocking
  [illegal_activity]=illegal  [harassment]=harassment  [hate]=hate
)

add_cell () {
  local D="$1" C="$2"
  local json="$D/categories_qwen3_vl_${C}_v5.json"
  if [ ! -f "$json" ]; then
    echo "${D}|${C}" >> $LIST
  fi
}

# ---- phase_paper_best: cell name = <base>_anchor or <base>_hybrid → rubric of <base>
for d in $BASE/outputs/phase_paper_best/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -gt 0 ] || continue
  base=${name%_anchor}; base=${base%_hybrid}
  rubric=${CELL2CONCEPT[$base]:-}
  if [ -z "$rubric" ]; then
    echo "[$(date)] WARN paper_best unknown $name" >> $LOGDIR/eval_v5_v2.log; continue
  fi
  add_cell "${d%/}" "$rubric"
done

# ---- phase_tune: <base>_tune_sh*_tau* → rubric of <base>
for d in $BASE/outputs/phase_tune/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -gt 0 ] || continue
  base=$(echo "$name" | sed -E 's/_tune_sh[^_]+_tau[^_]+$//')
  rubric=${CELL2CONCEPT[$base]:-}
  if [ -z "$rubric" ]; then
    echo "[$(date)] WARN tune unknown $name (base=$base)" >> $LOGDIR/eval_v5_v2.log; continue
  fi
  add_cell "${d%/}" "$rubric"
done

# ---- phase_selfharm: every cell → self_harm
for d in $BASE/outputs/phase_selfharm/*/; do
  [ -d "$d" ] || continue
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -gt 0 ] || continue
  add_cell "${d%/}" "self_harm"
done

# ---- phase_multi_v3: <group>__sh*_tau*__eval_<x> → rubric of <x>
for d in $BASE/outputs/phase_multi_v3/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(ls "$d"/*.png 2>/dev/null | wc -l)" -gt 0 ] || continue
  eval_part=${name##*__eval_}
  rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
  add_cell "${d%/}" "$rubric"
done

# ---- phase_safree: <group>__eval_<x> → rubric of <x> (PNGs directly under cell dir)
for d in $BASE/outputs/phase_safree/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  [ "$(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -gt 0 ] || continue
  eval_part=${name##*__eval_}
  rubric=${EVAL2RUBRIC[$eval_part]:-$eval_part}
  add_cell "${d%/}" "$rubric"
done

N=$(wc -l < $LIST)
echo "[$(date)] eval pending v2: $N cells" | tee -a $LOGDIR/eval_v5_v2.log

if [ "$N" -eq 0 ]; then
  echo "Nothing to evaluate. Done."
  exit 0
fi

# Round-robin across 8 GPUs.
NSLOTS=8
for slot in 0 1 2 3 4 5 6 7; do
  WLOG=$LOGDIR/eval_v5_v2_g${slot}.log
  > "$WLOG"
  (
    i=0
    while IFS='|' read -r D C; do
      if [ $((i % NSLOTS)) -eq $slot ]; then
        json="$D/categories_qwen3_vl_${C}_v5.json"
        if [ -f "$json" ]; then
          echo "[$(date)] [g$slot] SKIP $D $C" >> "$WLOG"
        else
          echo "[$(date)] [g$slot] EVAL $D $C" >> "$WLOG"
          cd $REPO/vlm
          CUDA_VISIBLE_DEVICES=$slot $PY $EVAL "$D" "$C" qwen >> "$WLOG" 2>&1
          rc=$?
          if [ $rc -ne 0 ]; then
            echo "[$(date)] [g$slot] FAIL $D $C exit=$rc" >> "$WLOG"
          else
            echo "[$(date)] [g$slot] DONE $D $C" >> "$WLOG"
          fi
        fi
      fi
      i=$((i+1))
    done < $LIST
    echo "[$(date)] [g$slot] worker complete" >> "$WLOG"
  ) &
done
wait
echo "[$(date)] all eval workers complete" | tee -a $LOGDIR/eval_v5_v2.log
