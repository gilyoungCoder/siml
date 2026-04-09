#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MASTER SCRIPT: ALL official baselines × ALL datasets
# Assign to GPUs on siml-01 and siml-02
# =============================================================================

SCRIPT=/mnt/home3/yhgil99/unlearning/scripts/run_sgf_official.sh
LOGDIR=/mnt/home3/yhgil99/unlearning/logs/official_baselines
mkdir -p $LOGDIR

# First symlink all missing datasets into SGF/nudity_sdv1/datasets/
SGFDS=/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1/datasets
REPO=/mnt/home3/yhgil99/unlearning
[ ! -f "$SGFDS/nudity-ring-a-bell.csv" ] && [ -f "$SGFDS/nudity-ring-a-bell.csv" ] || true
[ ! -f "$SGFDS/p4dn_16_prompt.csv" ] && ln -sf $REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_sexual.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_violence.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_violence.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_harassment.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_harassment.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_hate.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_hate.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_shocking.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_shocking.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_illegal_activity.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv $SGFDS/ 2>/dev/null || true
[ ! -f "$SGFDS/i2p_self-harm.csv" ] && ln -sf $REPO/SAFREE/datasets/i2p_categories/i2p_self-harm.csv $SGFDS/ 2>/dev/null || true

run() {
  local gpu=$1 method=$2 dataset=$3
  echo "[$(date +%H:%M)] Launching GPU $gpu: $method × $dataset"
  nohup bash $SCRIPT $gpu $dataset $method > $LOGDIR/${method}_${dataset}_gpu${gpu}.log 2>&1 &
}

# =============================================================================
# GPU ASSIGNMENTS (argument: gpu_list)
# =============================================================================

GPU_LIST=${1:-"0 2 3 5 6 7"}

GPUS=($GPU_LIST)
idx=0

# Methods × Datasets to run
# Priority 1: SGF + Safe_Denoiser (new methods not in workshop paper)
JOBS=(
  "sgf nudity_rab"
  "sgf nudity_mma"
  "sgf nudity_ud"
  "sgf nudity_p4dn"
  "sgf violence"
  "sgf harassment"
  "sgf hate"
  "sgf shocking"
  "safe_denoiser nudity_rab"
  "safe_denoiser nudity_mma"
  "safe_denoiser nudity_ud"
  "safe_denoiser nudity_p4dn"
  "safe_denoiser violence"
  "safe_denoiser harassment"
  "safe_denoiser hate"
  "safe_denoiser shocking"
  "vanilla nudity_rab"
  "vanilla nudity_mma"
  "vanilla nudity_ud"
  "vanilla nudity_p4dn"
  "vanilla violence"
  "vanilla harassment"
  "vanilla hate"
  "vanilla shocking"
)

# Run jobs round-robin across GPUs, sequential per GPU
declare -A GPU_JOBS
for gpu in "${GPUS[@]}"; do
  GPU_JOBS[$gpu]=""
done

for job in "${JOBS[@]}"; do
  method=$(echo $job | cut -d' ' -f1)
  dataset=$(echo $job | cut -d' ' -f2)
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  GPU_JOBS[$gpu]+="bash $SCRIPT $gpu $dataset $method 2>&1 | tail -3; "
  idx=$((idx + 1))
done

# Launch each GPU's job queue as a single background process
for gpu in "${GPUS[@]}"; do
  if [ -n "${GPU_JOBS[$gpu]}" ]; then
    echo "=== GPU $gpu: $(echo "${GPU_JOBS[$gpu]}" | grep -o "bash" | wc -l) jobs ==="
    nohup bash -c "${GPU_JOBS[$gpu]} echo GPU${gpu}_DONE" > $LOGDIR/gpu${gpu}_all.log 2>&1 &
  fi
done

echo "=============================================="
echo "  Official Baselines launched on GPUs: ${GPUS[*]}"
echo "  Methods: sgf, safe_denoiser, vanilla"
echo "  Datasets: nudity(rab,mma,ud,p4dn) + concepts(vio,har,hate,shock)"
echo "  Logs: $LOGDIR/"
echo "=============================================="

wait
echo "ALL DONE — $(date)"
