#!/bin/bash
set -uo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
LOG=$ROOT/logs/siml09_g${GPU}_ours_tune_eval.log
concept_arg() {
  case "$1" in
    sexual) echo nudity ;;
    self-harm) echo self_harm ;;
    illegal_activity) echo illegal ;;
    *) echo "$1" ;;
  esac
}
echo "[EVAL] start $(date) gpu=$GPU" | tee -a "$LOG"
for c in sexual violence self-harm shocking illegal_activity harassment hate; do
  CARG=$(concept_arg "$c")
  BASE="$ROOT/outputs/ours_tune/i2p_q16/$c"
  [ -d "$BASE" ] || continue
  for D in "$BASE"/*; do
    [ -d "$D" ] || continue
    N=$(find "$D" -maxdepth 1 -type f -name '*.png' | wc -l)
    if [ "$N" -lt 60 ]; then
      echo "[EVAL] SKIP incomplete $D n=$N" | tee -a "$LOG"
      continue
    fi
    if ls "$D"/categories_qwen3_vl_*_v5.json >/dev/null 2>&1; then
      echo "[EVAL] SKIP existing $D" | tee -a "$LOG"
      continue
    fi
    echo "[EVAL] RUN $D concept=$CARG n=$N $(date)" | tee -a "$LOG"
    cd /mnt/home3/yhgil99/unlearning/vlm
    CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$EVAL" "$D" "$CARG" qwen >> "$LOG" 2>&1
    echo "[EVAL] DONE rc=$? $D $(date)" | tee -a "$LOG"
  done
done
echo "[EVAL] all done $(date)" | tee -a "$LOG"
