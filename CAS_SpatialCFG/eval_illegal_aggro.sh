#!/bin/bash
# usage: eval_illegal_aggro.sh <worker_idx 0-14>
WORKER=$1
LAUNCH=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0423_illegal_aggro
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
LOGD=$LAUNCH/_eval_logs
mkdir -p $LOGD
ALL=($(ls -1d $LAUNCH/*/ | grep -v _eval_logs | sort))
N=${#ALL[@]}
if [ $WORKER -lt 5 ]; then
  START=$((WORKER * 3)); END=$((START + 3))
else
  START=$((15 + (WORKER - 5) * 2)); END=$((START + 2))
fi
echo "[w$WORKER] N=$N range=[$START,$END) host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES" | tee -a $LOGD/worker_$WORKER.log
for ((i=START; i<END && i<N; i++)); do
  D=${ALL[$i]}
  D=${D%/}
  R=$D/categories_qwen3_vl_illegal_v5.json
  if [ -f "$R" ]; then
    echo "[w$WORKER] SKIP $D" | tee -a $LOGD/worker_$WORKER.log
    continue
  fi
  echo "[w$WORKER] EVAL $D" | tee -a $LOGD/worker_$WORKER.log
  cd /mnt/home3/yhgil99/unlearning/vlm
  /mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10 $EVAL "$D" illegal qwen >> $LOGD/worker_$WORKER.log 2>&1
done
echo "[w$WORKER] DONE" | tee -a $LOGD/worker_$WORKER.log
