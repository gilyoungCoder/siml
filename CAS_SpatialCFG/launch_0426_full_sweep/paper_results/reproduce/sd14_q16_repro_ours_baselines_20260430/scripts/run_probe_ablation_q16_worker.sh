#!/usr/bin/env bash
set -euo pipefail
GPU=$1
QUEUE=$2
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
LOG=$ROOT/logs/probe_ablation_q16top60_20260501
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
mkdir -p "$LOG"
while IFS=, read -r PROBE CONCEPT EVALCONCEPT; do
  [ -n "$PROBE" ] || continue
  CFG=$ROOT/configs/probe_ablation_q16top60_20260501/$PROBE/$CONCEPT.json
  OUT=$ROOT/outputs/probe_ablation_q16top60_20260501/$PROBE/$CONCEPT
  RES=$OUT/results_qwen3_vl_${EVALCONCEPT}_v5.txt
  [ "$EVALCONCEPT" = nudity ] && RES=$OUT/results_qwen3_vl_nudity_v5.txt
  echo "[$(date)] GPU=$GPU START gen/eval probe=$PROBE concept=$CONCEPT eval=$EVALCONCEPT" | tee -a "$LOG/worker_gpu${GPU}.log"
  REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$GPU \
    "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$CFG" \
    2>&1 | tee -a "$LOG/${PROBE}_${CONCEPT}_gen_gpu${GPU}.log"
  N=$(find "$OUT" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "[$(date)] GPU=$GPU GEN_DONE probe=$PROBE concept=$CONCEPT n=$N" | tee -a "$LOG/worker_gpu${GPU}.log"
  if [ ! -s "$RES" ]; then
    CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$OUT" "$EVALCONCEPT" qwen \
      2>&1 | tee -a "$LOG/${PROBE}_${CONCEPT}_eval_gpu${GPU}.log"
  else
    echo "[$(date)] SKIP eval existing $RES" | tee -a "$LOG/worker_gpu${GPU}.log"
  fi
  echo "[$(date)] GPU=$GPU DONE probe=$PROBE concept=$CONCEPT" | tee -a "$LOG/worker_gpu${GPU}.log"
done < "$QUEUE"
