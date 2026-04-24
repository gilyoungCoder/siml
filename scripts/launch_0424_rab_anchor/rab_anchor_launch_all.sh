#!/bin/bash
# Launch all 8 RAB anchor chunks on siml-01 GPU 0-7 in parallel via nohup.
# 79 prompts → 8 shards: 10/10/10/10/10/10/10/9
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_rab_anchor/rab_anchor_v2pack_chunk.sh
LOGDIR=$REPO/logs/launch_0424_rab_anchor
mkdir -p $LOGDIR

# Shard ranges (start, end) — Python-style end-exclusive
RANGES=(
  "0 10"
  "10 20"
  "20 30"
  "30 40"
  "40 50"
  "50 60"
  "60 70"
  "70 79"
)

for GPU in 0 1 2 3 4 5 6 7; do
  range=${RANGES[$GPU]}
  read START END <<< "$range"
  nohup bash $SCRIPT $GPU $START $END > $LOGDIR/launch_g${GPU}.out 2>&1 &
  echo "Launched GPU=$GPU shard=$START-$END pid=$!"
done
wait
echo "[$(date)] All 8 chunks finished"
