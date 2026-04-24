#!/bin/bash
# COCO 10k generation for 3 modes (baseline / anchor / hybrid) across siml-01 g0-g7 + siml-02 g0-g7.
# 9966 prompts total. Distribute as 3 modes x ~3322 prompts each, split across 16 GPUs.
# Per-mode split: ~5-6 GPUs handle ~600 prompts each. Mix on each host.
# Strategy:
#   siml-01 g0-g4 (5 GPUs): baseline 10k split into 5 chunks of ~2000 each
#   siml-01 g5-g7 + siml-02 g0-g1 (5 GPUs): anchor 10k split
#   siml-02 g2-g7 (6 GPUs): hybrid 10k split  (NO — we want fairness)
# Better: each mode gets exactly 5-6 GPUs, sequential per host.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_coco10k/coco10k_gen_chunk.sh
N=9966   # total prompts in coco_10k.txt

# Plan: 3 modes x ~5-6 GPUs each.
#  baseline: siml-01 g0..g4 (5 GPUs)  -> chunks of ceil(N/5)=1994
#  anchor:   siml-01 g5..g7 + siml-02 g0..g1 (5 GPUs) -> chunks of 1994
#  hybrid:   siml-02 g2..g7 (6 GPUs) -> chunks of ceil(N/6)=1661
# Each chunk approx covers full 10k; ETA ~5-7 hr per cell.
declare -a JOBS

# baseline 5 chunks on siml-01 g0..4
for i in 0 1 2 3 4; do
  S=$((i*1994))
  E=$((S+1994)); [ $E -gt $N ] && E=$N
  JOBS+=("siml-01 $i baseline $S $E")
done

# anchor 5 chunks: siml-01 g5..7 + siml-02 g0..1
for i in 5 6 7; do
  idx=$((i-5))
  S=$((idx*1994))
  E=$((S+1994)); [ $E -gt $N ] && E=$N
  JOBS+=("siml-01 $i anchor $S $E")
done
for i in 0 1; do
  idx=$((i+3))
  S=$((idx*1994))
  E=$((S+1994)); [ $E -gt $N ] && E=$N
  JOBS+=("siml-02 $i anchor $S $E")
done

# hybrid 6 chunks on siml-02 g2..7
for i in 2 3 4 5 6 7; do
  idx=$((i-2))
  S=$((idx*1661))
  E=$((S+1661)); [ $E -gt $N ] && E=$N
  JOBS+=("siml-02 $i hybrid $S $E")
done

mkdir -p $REPO/logs/launch_0424_coco10k
ssh siml-01 "mkdir -p $REPO/logs/launch_0424_coco10k"
ssh siml-02 "mkdir -p $REPO/logs/launch_0424_coco10k"
for spec in "${JOBS[@]}"; do
  read HOST GPU MODE START END <<< "$spec"
  ssh $HOST "nohup bash $SCRIPT $GPU $MODE $START $END </dev/null >/dev/null 2>&1 & disown"
  echo "Launched $HOST g$GPU $MODE $START-$END"
done
echo "[$(date)] All 16 COCO 10k chunks dispatched (3 modes x 5-6 GPUs)"
