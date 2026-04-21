#!/bin/bash
# Urgent v5 eval for I2P top60 shocking/illegal/self-harm grid-A/B/sweep (batches pending).
# Usage: bash urgent_eval_3concepts.sh <gpu_id>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -x "$PY" ] || PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR
LOG="$LOGDIR/urgent_eval_g${GPU}.log"

declare -A CFILE
CFILE[shocking]=shocking
CFILE[illegal_activity]=illegal
CFILE[self-harm]=self_harm
CFILE[harassment]=harassment
CFILE[hate]=hate

cd $REPO/vlm
for concept in shocking illegal_activity self-harm harassment hate; do
  fc=${CFILE[$concept]}
  for src in ours_sd14_grid_v1pack ours_sd14_grid_v1pack_b ours_sd14_ablation_txtonly ours_sd14_ablation_imgonly; do
    base=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p/$src/$concept
    [ ! -d "$base" ] && continue
    for d in "$base"/*; do
      [ ! -d "$d" ] && continue
      n=$(ls -1 "$d"/*.png 2>/dev/null | wc -l)
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
echo "[$(date)] [g$GPU] urgent eval done"
