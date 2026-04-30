#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
while pgrep -af 'run_coco_official_chunk.sh' | grep -q safedenoiser; do sleep 30; done
cd "$ROOT"
scripts/run_runtime_benchmark_5methods.sh 0
