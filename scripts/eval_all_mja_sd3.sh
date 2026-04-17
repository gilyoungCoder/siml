#!/bin/bash
# $1=GPU, $2=target_dir, $3=concept
set -e
GPU=$1
TARGET_DIR=$2
CONCEPT=$3

export CUDA_VISIBLE_DEVICES=$GPU
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/batch_eval"
mkdir -p "$LOG"
cd /mnt/home3/yhgil99/unlearning/vlm

echo "[GPU$GPU $CONCEPT] START $(date) on $TARGET_DIR"

case "$CONCEPT" in
  nudity) rname="results_qwen3_vl_nudity.txt" ;;
  violence) rname="results_qwen3_vl_violence.txt" ;;
  shocking) rname="results_qwen3_vl_shocking.txt" ;;
  illegal) rname="results_qwen3_vl_illegal.txt" ;;
esac

count=0
for d in "$TARGET_DIR"/*/; do
  [ -d "$d" ] || continue
  [ $(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l) -eq 0 ] && continue
  [ -f "$d/$rname" ] && continue
  name=$(basename "$d" | head -c 50)
  $VLM_PY "$EVAL" "$d" "$CONCEPT" qwen > "$LOG/gpu${GPU}_${name}.log" 2>&1
  count=$((count+1))
done

echo "[GPU$GPU $CONCEPT] DONE $(date) — evaluated $count dirs"
