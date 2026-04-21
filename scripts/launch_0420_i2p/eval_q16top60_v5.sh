#!/bin/bash
# v5 eval on launch_0420_i2p_q16top60 outdirs.
# Usage: bash eval_q16top60_v5.sh <gpu_id>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PYTHON" ] || PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p_q16top60
mkdir -p $LOGDIR
LOG="$LOGDIR/eval_v5_g${GPU}.log"

declare -A CONCEPT_FILE
CONCEPT_FILE[violence]=violence
CONCEPT_FILE[self-harm]=self_harm
CONCEPT_FILE[shocking]=shocking
CONCEPT_FILE[illegal_activity]=illegal
CONCEPT_FILE[harassment]=harassment
CONCEPT_FILE[hate]=hate

cd $REPO/vlm
for concept in violence self-harm shocking illegal_activity harassment hate; do
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_q16top60/baseline_sd14/$concept"
  if [ ! -d "$outdir" ]; then continue; fi
  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  file_c="${CONCEPT_FILE[$concept]}"
  cat_v5="$outdir/categories_qwen3_vl_${file_c}_v5.json"
  if [ -f "$cat_v5" ]; then
    v5_count=$(python3 -c "import json; print(len(json.load(open('$cat_v5'))))" 2>/dev/null || echo 0)
    if [ "$v5_count" -ge "$n_imgs" ]; then
      echo "[$(date)] [g$GPU] [skip] $concept (v5 up-to-date)" | tee -a "$LOG"
      continue
    fi
  fi
  echo "[$(date)] [g$GPU] [v5] $concept n=$n_imgs" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v5.py "$outdir" "$file_c" qwen >> "$LOG" 2>&1 || echo "  FAILED $concept" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU] done"
