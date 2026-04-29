#!/bin/bash
# Nohup launch examples for tomorrow. This file is documentation + executable helper.
# It does NOT run unless you explicitly execute it.
set -euo pipefail
PLAN_ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd3_flux_i2p7_sweep_plan_20260429
OUT_ROOT=${OUT_ROOT:-/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_sd3_flux_i2p7_sweep}
mkdir -p "$OUT_ROOT/logs"

usage(){
  cat <<EOF
Usage:
  bash $0 pilot <sd3|flux1|both>
  bash $0 full  <sd3|flux1|both>

Examples:
  nohup bash $0 pilot both  > $OUT_ROOT/logs/launch_pilot_both.nohup.log 2>&1 &
  nohup bash $0 full  sd3   > $OUT_ROOT/logs/launch_full_sd3.nohup.log 2>&1 &

This launches 8 workers per selected model, one per GPU id 0..7.
EOF
}
MODE=${1:-}
WHICH=${2:-}
if [[ "$MODE" != "pilot" && "$MODE" != "full" ]]; then usage; exit 2; fi
if [[ "$WHICH" != "sd3" && "$WHICH" != "flux1" && "$WHICH" != "both" ]]; then usage; exit 2; fi

launch_model(){
  local model=$1
  for g in 0 1 2 3 4 5 6 7; do
    nohup bash "$PLAN_ROOT/run_matrix.sh" "$model" "$g" 8 "$g" "$MODE" \
      > "$OUT_ROOT/logs/${model}_${MODE}_g${g}.nohup.log" 2>&1 &
    echo $! > "$OUT_ROOT/logs/${model}_${MODE}_g${g}.pid"
    echo "launched $model $MODE gpu=$g pid=$(cat "$OUT_ROOT/logs/${model}_${MODE}_g${g}.pid")"
  done
}

if [[ "$WHICH" == "sd3" || "$WHICH" == "both" ]]; then launch_model sd3; fi
if [[ "$WHICH" == "flux1" || "$WHICH" == "both" ]]; then launch_model flux1; fi
