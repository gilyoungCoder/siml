#!/bin/bash
# Urgent v5 eval for ours_sd14_multi + safree_sd14_multi only.
# Usage: bash urgent_multi_eval.sh <gpu_id>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PY" ] || PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOG=$REPO/logs/launch_0420_i2p/urgent_multi_eval_g${GPU}.log
mkdir -p $REPO/logs/launch_0420_i2p

declare -A CFILE
CFILE[violence]=violence
CFILE[self-harm]=self_harm
CFILE[shocking]=shocking
CFILE[illegal_activity]=illegal
CFILE[harassment]=harassment
CFILE[hate]=hate

cd $REPO/vlm
for src in ours_sd14_multi safree_sd14_multi; do
  for concept in violence self-harm shocking illegal_activity harassment hate; do
    fc=${CFILE[$concept]}
    base=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/$src/$concept
    [ ! -d "$base" ] && continue
    # safree_multi has no cfg subdir; ours_multi has cfg subdir
    candidates=()
    if ls "$base"/*.png 1>/dev/null 2>&1; then candidates+=("$base"); fi
    for sub in "$base"/*/; do [ -d "$sub" ] && candidates+=("${sub%/}"); done
    for d in "${candidates[@]}"; do
      n=$(ls "$d"/*.png 2>/dev/null | wc -l)
      [ "$n" -lt 55 ] && continue
      jf="$d/categories_qwen3_vl_${fc}_v5.json"
      if [ -f "$jf" ]; then
        c=$(python3 -c "import json; print(len(json.load(open('$jf'))))" 2>/dev/null || echo 0)
        [ "$c" -ge "$n" ] && continue
      fi
      echo "[$(date)] [g$GPU] eval $src/$concept/$(basename $d) n=$n" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES=$GPU $PY opensource_vlm_i2p_all_v5.py "$d" "$fc" qwen >> "$LOG" 2>&1 || echo "FAIL $d" | tee -a "$LOG"
    done
  done
done
echo "[$(date)] [g$GPU] urgent multi eval done"
