#!/bin/bash
set -uo pipefail
DIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce
ROOT=$DIR/siml05_main_repro_20260429_0318
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR" "$ROOT/i2p60" "$ROOT/nudity"
echo "[$(date)] phaseA start on $(hostname)" | tee "$LOGDIR/phaseA.status"
(
  bash "$DIR/run_violence.sh" 0 "$ROOT/i2p60/violence" && \
  bash "$DIR/run_nudity_p4dn.sh" 0 "$ROOT/nudity/p4dn"
) > "$LOGDIR/gpu0_violence_then_p4dn.log" 2>&1 & p0=$!
(
  bash "$DIR/run_self-harm.sh" 1 "$ROOT/i2p60/self-harm"
) > "$LOGDIR/gpu1_self-harm.log" 2>&1 & p1=$!
(
  bash "$DIR/run_shocking.sh" 2 "$ROOT/i2p60/shocking"
) > "$LOGDIR/gpu2_shocking.log" 2>&1 & p2=$!
(
  bash "$DIR/run_illegal.sh" 3 "$ROOT/i2p60/illegal"
) > "$LOGDIR/gpu3_illegal.log" 2>&1 & p3=$!
(
  bash "$DIR/run_harassment.sh" 4 "$ROOT/i2p60/harassment"
) > "$LOGDIR/gpu4_harassment.log" 2>&1 & p4=$!
(
  bash "$DIR/run_hate.sh" 5 "$ROOT/i2p60/hate"
) > "$LOGDIR/gpu5_hate.log" 2>&1 & p5=$!
(
  bash "$DIR/run_nudity_ud.sh" 6 "$ROOT/nudity/ud"
) > "$LOGDIR/gpu6_nudity_ud.log" 2>&1 & p6=$!
(
  bash "$DIR/run_nudity_rab.sh" 7 "$ROOT/nudity/rab"
) > "$LOGDIR/gpu7_nudity_rab.log" 2>&1 & p7=$!
rc=0
for p in $p0 $p1 $p2 $p3 $p4 $p5 $p6 $p7; do
  wait "$p" || rc=$?
done
echo "[$(date)] phaseA done rc=$rc" | tee -a "$LOGDIR/phaseA.status"
exit $rc
