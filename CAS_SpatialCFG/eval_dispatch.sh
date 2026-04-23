#!/bin/bash
# usage: eval_dispatch.sh <worker_idx 0-14>
WORKER=$1
LIST=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/eval_pending.txt
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOGD=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/eval_dispatch_logs
mkdir -p $LOGD
echo "[w$WORKER] start host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES list=$(wc -l < $LIST)" >> $LOGD/worker_$WORKER.log
i=0
while IFS='|' read -r D C; do
  if [ $((i % 15)) -eq $WORKER ]; then
    R=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/$D/categories_qwen3_vl_${C}_v5.json
    if [ -f "$R" ]; then
      echo "[w$WORKER] SKIP $D $C" >> $LOGD/worker_$WORKER.log
    else
      echo "[w$WORKER] EVAL $D $C" >> $LOGD/worker_$WORKER.log
      cd /mnt/home3/yhgil99/unlearning/vlm
      $PY $EVAL /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/$D $C qwen >> $LOGD/worker_$WORKER.log 2>&1
    fi
  fi
  i=$((i+1))
done < $LIST
echo "[w$WORKER] DONE" >> $LOGD/worker_$WORKER.log
