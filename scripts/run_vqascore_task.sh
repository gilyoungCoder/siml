#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?gpu}
IMGDIR=${2:?imgdir}
PROMPTS=${3:?prompts}
MODE=${4:-simple}

export PYTHONNOUSERSITE=1
P=/mnt/home3/yhgil99/.conda/envs/vqascore/bin/python3.10
REPO=/mnt/home3/yhgil99/unlearning
cd "$REPO"

if [ "$MODE" = alignment ]; then
  OUT="$IMGDIR/results_vqascore_alignment.json"
  [ -f "$OUT" ] && echo "[SKIP] $IMGDIR alignment" && exit 0
  CUDA_VISIBLE_DEVICES=$GPU "$P" "$REPO/vlm/eval_vqascore_alignment.py" "$IMGDIR" --prompts "$PROMPTS"
else
  OUT="$IMGDIR/results_vqascore.json"
  [ -f "$OUT" ] && echo "[SKIP] $IMGDIR simple" && exit 0
  CUDA_VISIBLE_DEVICES=$GPU "$P" "$REPO/vlm/eval_vqascore.py" "$IMGDIR" --prompts "$PROMPTS"
fi
