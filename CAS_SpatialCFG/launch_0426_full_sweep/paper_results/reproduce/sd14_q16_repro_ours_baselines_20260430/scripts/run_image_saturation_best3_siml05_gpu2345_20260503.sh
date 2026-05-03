#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAND=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduction_for_paper_20260501_REVIEWER_READY_CANDIDATE
PYGEN=${PYGEN:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
PYVLM=${PYVLM:-/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10}
V5=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
EXP=image_saturation_best3_finalconfig_seed42_20260503
CFGBASE=$ROOT/configs/$EXP
OUTBASE=$ROOT/outputs/$EXP
LOG=$ROOT/logs/${EXP}_siml05_gpu2345
LOCK=$LOG/locks
mkdir -p "$LOG" "$LOCK"
SEED=42
eval_concept(){ case "$1" in sexual) echo nudity;; illegal_activity) echo illegal;; self-harm) echo self_harm;; *) echo "$1";; esac; }
run_one(){
  local gpu=$1 c=$2 k=$3 evalc cfg out res lockdir
  evalc=$(eval_concept "$c")
  cfg=$CFGBASE/$c/k${k}/seed${SEED}.json
  out=$OUTBASE/$c/k${k}/seed${SEED}
  lockdir=$LOCK/${c}_k${k}.lock
  if ! mkdir "$lockdir" 2>/dev/null; then echo "[$(date)] GPU=$gpu LOCKED c=$c k=$k" | tee -a "$LOG/worker_gpu${gpu}.log"; return 0; fi
  res=$(ls "$out"/results_qwen3_vl_*_v5.txt 2>/dev/null | head -1 || true)
  if [ -s "$res" ]; then echo "[$(date)] GPU=$gpu SKIP c=$c k=$k has_result=$res" | tee -a "$LOG/worker_gpu${gpu}.log"; rmdir "$lockdir" 2>/dev/null || true; return 0; fi
  echo "[$(date)] GPU=$gpu START c=$c k=$k eval=$evalc" | tee -a "$LOG/worker_gpu${gpu}.log"
  if [ "$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)" -lt 60 ]; then
    REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$gpu "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$gpu" --config "$cfg" --force 2>&1 | tee -a "$LOG/${c}_k${k}_gen_gpu${gpu}.log"
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${c}_k${k}_eval_gpu${gpu}.log"
  echo "[$(date)] GPU=$gpu DONE c=$c k=$k" | tee -a "$LOG/worker_gpu${gpu}.log"
  rmdir "$lockdir" 2>/dev/null || true
}
worker(){
  local gpu=$1 queue=$2
  while IFS=, read -r c k; do [ -n "$c" ] && run_one "$gpu" "$c" "$k"; done < "$queue"
}
prepare_queues(){
  : > "$LOG/queue_gpu2.csv"; : > "$LOG/queue_gpu3.csv"; : > "$LOG/queue_gpu4.csv"; : > "$LOG/queue_gpu5.csv"
  printf "hate,1\nhate,2\nhate,4\n" > "$LOG/queue_gpu2.csv"
  printf "hate,8\nhate,12\nhate,16\n" > "$LOG/queue_gpu3.csv"
  printf "violence,1\nviolence,2\nviolence,4\n" > "$LOG/queue_gpu4.csv"
  printf "violence,8\nviolence,12\nviolence,16\n" > "$LOG/queue_gpu5.csv"
}
launch(){
  prepare_queues
  for gpu in 2 3 4 5; do
    nohup bash "$0" worker "$gpu" "$LOG/queue_gpu${gpu}.csv" > "$LOG/launch_gpu${gpu}.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu${gpu}.pid"
    echo "launched gpu=$gpu pid=$(cat "$LOG/launch_gpu${gpu}.pid")"
  done
}
status(){
  echo "LOG=$LOG OUT=$OUTBASE"
  pgrep -af "run_image_saturation_best3_siml05_gpu2345|run_from_config|generate_family|opensource_vlm_i2p_all_v5" || true
  for c in hate violence; do for k in 1 2 4 8 12 16; do out=$OUTBASE/$c/k${k}/seed${SEED}; echo "$c k=$k png=$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) res=$(ls "$out"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done; done
}
case "${1:-launch}" in
  launch) launch;;
  worker) worker "$2" "$3";;
  status) status;;
  *) echo "Usage: $0 {launch|worker GPU QUEUE|status}" >&2; exit 2;;
esac
