#!/bin/bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
mkdir -p "$ROOT/logs" "$ROOT/pids"
launch(){ local gpu=$1 list=$2 name=$3; nohup "$ROOT/scripts/worker.sh" "$gpu" "$ROOT/joblists/$list" > "$ROOT/logs/${name}.log" 2>&1 & echo $! > "$ROOT/pids/${name}.pid"; echo "launched $name gpu=$gpu pid=$(cat "$ROOT/pids/${name}.pid")"; }
launch 1 siml01_g1_ours_i2p_a.tsv siml01_g1_ours_i2p_a
launch 2 siml01_g2_ours_i2p_b.tsv siml01_g2_ours_i2p_b
launch 3 siml01_g3_ours_i2p_c.tsv siml01_g3_ours_i2p_c
launch 4 siml01_g4_ours_nudity_a.tsv siml01_g4_ours_nudity_a
launch 5 siml01_g5_ours_nudity_p4dn.tsv siml01_g5_ours_nudity_p4dn
launch 6 siml01_g6_ours_nudity_mma.tsv siml01_g6_ours_nudity_mma
