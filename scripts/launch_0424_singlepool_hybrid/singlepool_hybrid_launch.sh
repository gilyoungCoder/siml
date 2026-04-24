#!/bin/bash
# Launch all 3 missing hybrid single-pool cells (sexual/illegal/disturbing)
# Distribute: 100 prompts × 3 cells across siml-02 g0-7 + siml-01 g5-g7 = 11 GPUs.
# Per-cell split: 3-4 GPUs each, ~25-33 prompts per GPU.
set -uo pipefail
REPO=/mnt/home3/yhgil99/unlearning
SCRIPT=$REPO/scripts/launch_0424_singlepool_hybrid/singlepool_hybrid_chunk.sh

# Spec: hostname GPU concept start end
JOBS=(
  # sexual: 4 chunks on siml-02 g0-g3 (25 each)
  "siml-02 0 sexual 0 25"
  "siml-02 1 sexual 25 50"
  "siml-02 2 sexual 50 75"
  "siml-02 3 sexual 75 100"
  # illegal: 4 chunks on siml-02 g4-g7
  "siml-02 4 illegal 0 25"
  "siml-02 5 illegal 25 50"
  "siml-02 6 illegal 50 75"
  "siml-02 7 illegal 75 100"
  # disturbing: 3 chunks on siml-01 g5-g7
  "siml-01 5 disturbing 0 34"
  "siml-01 6 disturbing 34 67"
  "siml-01 7 disturbing 67 100"
)

for spec in "${JOBS[@]}"; do
  read HOST GPU CONCEPT START END <<< "$spec"
  ssh $HOST "mkdir -p $REPO/logs/launch_0424_singlepool_hybrid; nohup bash $SCRIPT $GPU $CONCEPT $START $END </dev/null >/dev/null 2>&1 & disown"
  echo "Launched $HOST g$GPU $CONCEPT $START-$END"
done

echo "[$(date)] All 11 chunks dispatched"
