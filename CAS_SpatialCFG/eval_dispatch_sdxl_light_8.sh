#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=<gpu> eval_dispatch_sdxl_light_8.sh <worker_idx 0..7> [num_workers]
set -u
WORKER=${1:?worker index required}
NWORKERS=${2:-8}
LIST=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/eval_pending_sdxl_light.txt
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
OUTROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs
LOGD=/mnt/home3/yhgil99/unlearning/logs/launch_0425_sdxl_light_v5eval_8gpu
mkdir -p "$LOGD"
log="$LOGD/worker_${WORKER}.log"

norm_concept() {
  case "$1" in
    sexual) echo nudity ;;
    illegal_activity) echo illegal ;;
    self_harm) echo self-harm ;;
    *) echo "$1" ;;
  esac
}

result_name() {
  local c="$1"
  echo "categories_qwen3_vl_${c}_v5.json"
}

echo "[w$WORKER] start ts=$(date -Is) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} nworkers=$NWORKERS list=$(wc -l < "$LIST")" >> "$log"
i=0
while IFS="|" read -r D C; do
  [ -z "${D:-}" ] && continue
  if [ $((i % NWORKERS)) -eq "$WORKER" ]; then
    EC=$(norm_concept "$C")
    DIR="$OUTROOT/$D"
    R="$DIR/$(result_name "$EC")"
    if [ ! -d "$DIR" ]; then
      echo "[w$WORKER] MISSING_DIR $D $C eval=$EC" >> "$log"
    elif [ -f "$R" ]; then
      echo "[w$WORKER] SKIP $D original=$C eval=$EC result=$R" >> "$log"
    else
      echo "[w$WORKER] EVAL $D original=$C eval=$EC start=$(date -Is)" >> "$log"
      cd /mnt/home3/yhgil99/unlearning/vlm || exit 2
      "$PY" "$EVAL" "$DIR" "$EC" qwen >> "$log" 2>&1
      code=$?
      echo "[w$WORKER] DONE_ITEM $D eval=$EC code=$code end=$(date -Is)" >> "$log"
    fi
  fi
  i=$((i+1))
done < "$LIST"
echo "[w$WORKER] DONE_ALL ts=$(date -Is)" >> "$log"
