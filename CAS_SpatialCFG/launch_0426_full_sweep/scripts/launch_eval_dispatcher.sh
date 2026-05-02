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

# Auto-pick free GPUs on this host (skip any with >2GB used).
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 < 2000 {print $1}')
NWORKERS=${1:-${#FREE[@]}}
GPUS=("${FREE[@]:0:$NWORKERS}")
[ ${#GPUS[@]} -eq 0 ] && { echo "No free GPUs on this host"; exit 1; }

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
