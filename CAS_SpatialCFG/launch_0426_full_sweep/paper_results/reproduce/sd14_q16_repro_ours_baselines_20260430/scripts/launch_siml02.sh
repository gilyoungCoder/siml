#!/bin/bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
mkdir -p "$ROOT/logs" "$ROOT/pids"
launch(){ local gpu=$1 list=$2 name=$3; nohup "$ROOT/scripts/worker.sh" "$gpu" "$ROOT/joblists/$list" > "$ROOT/logs/${name}.log" 2>&1 & echo $! > "$ROOT/pids/${name}.pid"; echo "launched $name gpu=$gpu pid=$(cat "$ROOT/pids/${name}.pid")"; }
launch 1 siml02_g1_baseline_all.tsv siml02_g1_baseline_all
launch 2 siml02_g2_safree_a.tsv siml02_g2_safree_a
launch 4 siml02_g4_safree_b.tsv siml02_g4_safree_b
launch 5 siml02_g5_safedenoiser_a.tsv siml02_g5_safedenoiser_a
launch 6 siml02_g6_safedenoiser_b.tsv siml02_g6_safedenoiser_b
launch 7 siml02_g7_sgf_all.tsv siml02_g7_sgf_all
