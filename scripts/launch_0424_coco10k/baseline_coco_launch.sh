#!/bin/bash
# Launch baseline SD1.4 COCO 10k across siml-01 g0-g7 + siml-02 g0-g7 (16 GPUs).
# 9966 prompts / 16 GPUs = 623 prompts each.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_coco10k/baseline_coco_chunk.sh

idx=0
for HOST in siml-01 siml-02; do
  for GPU in 0 1 2 3 4 5 6 7; do
    START=$((idx * 623))
    END=$((START + 623))
    if [ $idx -eq 15 ]; then END=9966; fi
    ssh $HOST "nohup bash $SCRIPT $GPU $START $END </dev/null >/dev/null 2>&1 & disown"
    echo "Launched $HOST g$GPU baseline $START-$END"
    idx=$((idx+1))
  done
done
echo "[$(date)] All 16 baseline COCO 10k chunks dispatched"
