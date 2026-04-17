#!/bin/bash
# Batch eval — evaluates all directories in a list sequentially on given GPU
# $1=GPU, $2=concept (nudity/violence/shocking/illegal), $3+=directories
set -e
GPU=$1
CONCEPT=$2
shift 2

export CUDA_VISIBLE_DEVICES=$GPU
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/batch_eval"
mkdir -p "$LOG"
cd /mnt/home3/yhgil99/unlearning/vlm

echo "[GPU$GPU $CONCEPT] START $(date) — $# dirs"

for d in "$@"; do
  [ -d "$d" ] || continue
  [ $(find "$d" -name '*.png' 2>/dev/null | wc -l) -eq 0 ] && continue
  # Skip if already has result
  case "$CONCEPT" in
    nudity) result_name="results_qwen3_vl_nudity.txt" ;;
    violence) result_name="results_qwen3_vl_violence.txt" ;;
    shocking) result_name="results_qwen3_vl_shocking.txt" ;;
    illegal) result_name="results_qwen3_vl_illegal.txt" ;;
  esac
  [ -f "$d/$result_name" ] && continue

  name=$(basename "$d" | head -c 50)
  $VLM_PY "$EVAL" "$d" "$CONCEPT" qwen > "$LOG/${name}.log" 2>&1
done

echo "[GPU$GPU $CONCEPT] DONE $(date)"
