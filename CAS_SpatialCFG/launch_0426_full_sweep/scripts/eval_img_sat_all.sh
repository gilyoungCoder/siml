#!/bin/bash
# Parallel eval of all phase_img_sat_all cells.
# Args: $1 = GPU index, $2 = worker_id (0..N-1), $3 = total_workers
set -uo pipefail
GPU=${1:-0}
WID=${2:-0}
NWORK=${3:-5}
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOG=$BASE/logs/eval_img_sat_all_g${GPU}_w${WID}_$(date +%m%d_%H%M).log
> $LOG

# Cell list: (concept, K, rubric)
CELLS=(
  "violence|1|violence" "violence|2|violence" "violence|4|violence" "violence|8|violence" "violence|12|violence" "violence|16|violence"
  "illegal_activity|1|illegal" "illegal_activity|2|illegal" "illegal_activity|4|illegal" "illegal_activity|8|illegal" "illegal_activity|12|illegal" "illegal_activity|16|illegal"
  "shocking|1|shocking" "shocking|2|shocking" "shocking|4|shocking" "shocking|8|shocking" "shocking|12|shocking" "shocking|16|shocking"
  "harassment|1|harassment" "harassment|2|harassment" "harassment|4|harassment" "harassment|8|harassment" "harassment|12|harassment" "harassment|16|harassment"
  "hate|1|hate" "hate|2|hate" "hate|4|hate" "hate|8|hate" "hate|12|hate" "hate|16|hate"
  "self-harm|1|self_harm" "self-harm|2|self_harm" "self-harm|4|self_harm" "self-harm|8|self_harm" "self-harm|12|self_harm" "self-harm|16|self_harm"
  "sexual|12|nudity"
)

cd /mnt/home3/yhgil99/unlearning/vlm
i=0
for entry in "${CELLS[@]}"; do
  if [ $((i % NWORK)) -eq $WID ]; then
    IFS='|' read -r C K R <<< "$entry"
    D=$BASE/outputs/phase_img_sat_all/${C}_K${K}
    J=$D/categories_qwen3_vl_${R}_v5.json
    if [ -f "$J" ]; then
      echo "[$(date +%H:%M:%S)] [$WID] SKIP ${C}_K${K} (json exists)" | tee -a $LOG
    else
      echo "[$(date +%H:%M:%S)] [$WID] EVAL ${C}_K${K} rubric=$R" | tee -a $LOG
      CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL "$D" "$R" qwen >> $LOG 2>&1
      echo "[$(date +%H:%M:%S)] [$WID] DONE ${C}_K${K}" | tee -a $LOG
    fi
  fi
  i=$((i+1))
done
echo "[$(date)] worker $WID done" | tee -a $LOG
