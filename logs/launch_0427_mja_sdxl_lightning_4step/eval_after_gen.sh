#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/home3/yhgil99/unlearning
OUT=$BASE/CAS_SpatialCFG/outputs/launch_0427_mja_sdxl_lightning_4step
LOGD=$BASE/logs/launch_0427_mja_sdxl_lightning_4step
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$BASE/vlm/opensource_vlm_i2p_all_v5.py
# Wait for generation pids to finish
for pf in "$LOGD"/*_gpu*.pid; do
  pid=$(cat "$pf")
  echo "[$(date -Is)] waiting gen pid=$pid pf=$pf" >> "$LOGD/eval_after_gen.log"
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  echo "[$(date -Is)] gen done pid=$pid" >> "$LOGD/eval_after_gen.log"
done
# Evaluate with Qwen v5, one concept per GPU
run_eval(){
  local gpu=$1 dir=$2 concept=$3
  echo "[$(date -Is)] launch eval dir=$dir concept=$concept gpu=$gpu" | tee -a "$LOGD/eval_after_gen.log"
  (cd "$BASE/vlm" && CUDA_VISIBLE_DEVICES=$gpu "$PY" "$EVAL" "$OUT/$dir" "$concept" qwen > "$LOGD/eval_${dir}_${concept}_gpu${gpu}.log" 2>&1; echo "[$(date -Is)] eval done dir=$dir concept=$concept rc=$?" >> "$LOGD/eval_after_gen.log") &
}
run_eval 1 mja_violent violence
run_eval 2 mja_illegal illegal
run_eval 3 mja_disturbing disturbing
wait
# Summarize
for d in mja_violent mja_illegal mja_disturbing; do
  echo "--- $d" >> "$LOGD/eval_after_gen.log"
  ls -1 "$OUT/$d"/categories_qwen3_vl_*_v5.json "$OUT/$d"/results_qwen3_vl_*_v5.txt 2>/dev/null >> "$LOGD/eval_after_gen.log" || true
  find "$OUT/$d" -maxdepth 1 -type f -name "*.png" | wc -l >> "$LOGD/eval_after_gen.log"
done
