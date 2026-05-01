#!/bin/bash
# Eval random multi-seed K cells (3 concept × 2 K × 3 seed = 18). Skip if already evaled.
set -uo pipefail
GPU=${1:-6}
WID=${2:-0}
NWORK=${3:-1}
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOG=$BASE/logs/eval_random_K_g${GPU}_w${WID}_$(date +%m%d_%H%M).log
> $LOG

declare -A R=([violence]=violence [sexual]=nudity [hate]=hate)
CELLS=()
for C in violence sexual hate; do
  for K in 1 2; do
    for S in 42 43 44; do
      CELLS+=("$C|$K|$S|${R[$C]}")
    done
  done
done

cd /mnt/home3/yhgil99/unlearning/vlm
i=0
for entry in "${CELLS[@]}"; do
  if [ $((i % NWORK)) -eq $WID ]; then
    IFS='|' read -r C K S RUB <<< "$entry"
    D=$BASE/outputs/phase_img_sat_random/${C}_K${K}_seed${S}
    J=$D/categories_qwen3_vl_${RUB}_v5.json
    PNG=$(ls $D/*.png 2>/dev/null | wc -l)
    if [ "$PNG" -lt 60 ]; then
      echo "[$(date +%H:%M:%S)] [w$WID] WAIT ${C}_K${K}_s${S} ($PNG/60)" | tee -a $LOG
      # wait until 60 imgs (gen still running)
      while [ "$(ls $D/*.png 2>/dev/null | wc -l)" -lt 60 ]; do sleep 10; done
    fi
    if [ -f "$J" ]; then
      echo "[$(date +%H:%M:%S)] [w$WID] SKIP ${C}_K${K}_s${S}" | tee -a $LOG
    else
      echo "[$(date +%H:%M:%S)] [w$WID] EVAL ${C}_K${K}_s${S}" | tee -a $LOG
      CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL "$D" "$RUB" qwen >> $LOG 2>&1
      echo "[$(date +%H:%M:%S)] [w$WID] DONE ${C}_K${K}_s${S}" | tee -a $LOG
    fi
  fi
  i=$((i+1))
done
echo "[$(date)] worker $WID done" | tee -a $LOG
