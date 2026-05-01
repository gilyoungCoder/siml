#!/bin/bash
# Parallel v5 eval for phase_nfe_safedenoiser_sgf cells.
# Args: $1=GPU $2=WID $3=NWORK
set -uo pipefail
GPU=${1:-0}
WID=${2:-0}
NWORK=${3:-5}
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
LOG=$BASE/logs/eval_nfe_sdsgf_g${GPU}_w${WID}_$(date +%m%d_%H%M).log
> $LOG

declare -A R=([sexual]=nudity [violence]=violence [self-harm]=self_harm [shocking]=shocking)

CELLS=()
for METHOD in safedenoiser sgf; do
  for CONCEPT in violence shocking self-harm sexual; do
    for STEP in 1 3 5 8 12 16 20 25 30 40 50; do
      CELLS+=("$METHOD|$CONCEPT|$STEP")
    done
  done
done

cd /mnt/home3/yhgil99/unlearning/vlm
i=0
for entry in "${CELLS[@]}"; do
  if [ $((i % NWORK)) -eq $WID ]; then
    IFS='|' read -r M C S <<< "$entry"
    D=$BASE/outputs/phase_nfe_safedenoiser_sgf/${M}_${C}_step${S}/all
    rubric=${R[$C]}
    J=$D/categories_qwen3_vl_${rubric}_v5.json
    if [ -f "$J" ]; then
      echo "[$(date +%H:%M:%S)] [w$WID] SKIP ${M}/${C}/step${S}" | tee -a $LOG
    else
      n=$(ls $D/*.png 2>/dev/null | wc -l)
      if [ "$n" -lt 60 ]; then
        echo "[$(date +%H:%M:%S)] [w$WID] SKIP_INC ${M}/${C}/step${S} only $n/60" | tee -a $LOG
      else
        echo "[$(date +%H:%M:%S)] [w$WID] EVAL ${M}/${C}/step${S}" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL "$D" "$rubric" qwen >> $LOG 2>&1
        echo "[$(date +%H:%M:%S)] [w$WID] DONE ${M}/${C}/step${S}" | tee -a $LOG
      fi
    fi
  fi
  i=$((i+1))
done
echo "[$(date)] worker $WID done" | tee -a $LOG
