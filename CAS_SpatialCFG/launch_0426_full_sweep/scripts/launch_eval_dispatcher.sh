#!/usr/bin/env bash
# Eval dispatcher (run on siml-09): scans both phase_nfe_walltime_v3 and
# phase_scale_robustness for cells with PNGs but no Qwen v5 result file,
# then runs eval distributed across siml-09 GPUs.
#
# Usage: bash launch_eval_dispatcher.sh [NWORKERS]
#   NWORKERS = number of parallel eval slots (default = number of free GPUs).

set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs/eval_dispatcher
mkdir -p $LOGDIR

DISPATCHER=$SCRIPTS/eval_dispatcher.py
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
[ -f "$DISPATCHER" ] || { echo "Missing $DISPATCHER"; exit 1; }

# Pick GPUs that have >=20GB free memory (Qwen3-VL-8B fits in ~17GB).
# On big-memory cards (siml-09 g0 = ~97GB), launch multiple workers on the same GPU.
mapfile -t USABLE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' '$2 >= 20000 {print $1","$2}')
[ ${#USABLE[@]} -eq 0 ] && { echo "No GPU with >=20GB free on this host"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; exit 1; }

# Build worker list: one worker per ~20GB of free memory, up to user-specified cap.
GPUS=()
for entry in "${USABLE[@]}"; do
  IFS=',' read -r idx free <<< "$entry"
  slots_for_this_gpu=$(( free / 20000 ))
  [ $slots_for_this_gpu -gt 4 ] && slots_for_this_gpu=4   # cap at 4 workers per GPU
  for ((j=0; j<slots_for_this_gpu; j++)); do GPUS+=("$idx"); done
done

NWORKERS_CAP=${1:-${#GPUS[@]}}
GPUS=("${GPUS[@]:0:$NWORKERS_CAP}")
[ ${#GPUS[@]} -eq 0 ] && { echo "No worker slots available"; exit 1; }

echo "[$(date)] dispatching eval across $(hostname) GPUs ${GPUS[*]}"
for IDX in "${!GPUS[@]}"; do
  GPU=${GPUS[$IDX]}; WID=$IDX
  LOG=$LOGDIR/eval_g${GPU}_w${WID}.log
  nohup $PY $DISPATCHER $GPU $WID ${#GPUS[@]} >> $LOG 2>&1 &
  echo "  -> GPU=$GPU worker=$WID pid=$! log=$LOG"
done

echo
echo "Monitor:"
echo "  find $ROOT/outputs/phase_nfe_walltime_v3 -name 'results_qwen3_vl_*_v5.txt' -size +50c | wc -l"
echo "  find $ROOT/outputs/phase_scale_robustness -name 'results_qwen3_vl_*_v5.txt' -size +50c | wc -l"
echo "  tail -F $LOGDIR/*.log"
