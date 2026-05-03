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
LOG=$ROOT/logs/${EXP}_siml05_gpu5
mkdir -p "$LOG"
GPU=5
CONCEPTS=(hate violence)
K_LIST=(1 2 4 8 12 16)
SEED=42
eval_concept(){ case "$1" in sexual) echo nudity;; illegal_activity) echo illegal;; self-harm) echo self_harm;; *) echo "$1";; esac; }
run_one(){
  local c=$1 k=$2 evalc cfg out res
  evalc=$(eval_concept "$c")
  cfg=$CFGBASE/$c/k${k}/seed${SEED}.json
  out=$OUTBASE/$c/k${k}/seed${SEED}
  res=$(ls "$out"/results_qwen3_vl_*_v5.txt 2>/dev/null | head -1 || true)
  if [ -s "$res" ]; then
    echo "[$(date)] SKIP c=$c k=$k has_result=$res" | tee -a "$LOG/worker_gpu5.log"
    return 0
  fi
  echo "[$(date)] START siml05 GPU=5 c=$c k=$k eval=$evalc" | tee -a "$LOG/worker_gpu5.log"
  if [ "$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)" -lt 60 ]; then
    REPRO_ROOT=$CAND OUT_ROOT=$ROOT PY_SAFGEN=$PYGEN CUDA_VISIBLE_DEVICES=$GPU "$PYGEN" "$CAND/scripts/run_from_config.py" --gpu "$GPU" --config "$cfg" --force 2>&1 | tee -a "$LOG/${c}_k${k}_gen.log"
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PYVLM" "$V5" "$out" "$evalc" qwen 2>&1 | tee -a "$LOG/${c}_k${k}_eval.log"
  echo "[$(date)] DONE siml05 GPU=5 c=$c k=$k" | tee -a "$LOG/worker_gpu5.log"
}
case "${1:-worker}" in
  worker)
    for c in "${CONCEPTS[@]}"; do for k in "${K_LIST[@]}"; do run_one "$c" "$k"; done; done
    ;;
  launch)
    nohup bash "$0" worker > "$LOG/launch_gpu5.nohup.log" 2>&1 & echo $! > "$LOG/launch_gpu5.pid"; echo "launched siml05 gpu5 pid=$(cat "$LOG/launch_gpu5.pid")";;
  status)
    echo "LOG=$LOG OUT=$OUTBASE CFG=$CFGBASE"
    pgrep -af "run_image_saturation_best3_siml05|run_from_config|generate_family|opensource_vlm_i2p_all_v5" || true
    for c in "${CONCEPTS[@]}"; do for k in "${K_LIST[@]}"; do out=$OUTBASE/$c/k${k}/seed${SEED}; echo "$c k=$k png=$(find "$out" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) res=$(ls "$out"/results_qwen3_vl_*_v5.txt 2>/dev/null | wc -l)"; done; done
    ;;
  *) echo "Usage: $0 {launch|worker|status}" >&2; exit 2;;
esac
