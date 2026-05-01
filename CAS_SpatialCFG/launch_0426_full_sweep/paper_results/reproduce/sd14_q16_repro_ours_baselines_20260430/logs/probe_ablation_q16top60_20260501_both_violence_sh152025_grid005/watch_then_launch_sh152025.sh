#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
OLD=probe_ablation_q16top60_20260501_both_violence_adaptive_clean005
LOGDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_sh152025_grid005
OLDRES=$ROOT/outputs/$OLD/sh20_tau04_txt030_img010/results_qwen3_vl_violence_v5.txt
while [ ! -s "$OLDRES" ]; do
  echo "[$(date)] waiting current sh20 clean005 result before switching to sh15/20/25-only queue" >> "$LOGDIR/watcher.log"
  sleep 30
done
echo "[$(date)] old sh20 result exists; killing old clean005 queue and launching sh15/20/25-only" >> "$LOGDIR/watcher.log"
pids=$(pgrep -f "$ROOT/logs/$OLD/run_clean005_4567.sh|$ROOT/configs/$OLD|$ROOT/outputs/$OLD" || true)
[ -n "$pids" ] && kill $pids 2>/dev/null || true
sleep 2
pids=$(pgrep -f "$ROOT/logs/$OLD/run_clean005_4567.sh|$ROOT/configs/$OLD|$ROOT/outputs/$OLD" || true)
[ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
nohup "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_sh152025_grid005/run_sh152025_4567.sh" > "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/logs/probe_ablation_q16top60_20260501_both_violence_sh152025_grid005/nohup_sh152025_4567.log" 2>&1 &
echo "[$(date)] launched PID=$!" >> "$LOGDIR/watcher.log"
