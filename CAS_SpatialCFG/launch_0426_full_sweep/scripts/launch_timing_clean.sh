#!/usr/bin/env bash
# Launch CLEAN timing on siml-05 g0,g1,g4,g5: 1 method per GPU at a time, sequential within GPU.
# Distribution (load-balanced for ~32 min wall):
#   g0: baseline + safedenoiser + sld_strong   (~28 min)
#   g1: safree + sld_weak                       (~23 min)
#   g4: sgf + sld_medium                        (~23 min)
#   g5: sld_max + ebsg                          (~32 min)
# After all done, merge per-method CSVs into nfe_walltime_timing.csv (overwrite).
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs
mkdir -p $LOGDIR

launch_gpu_chain () {
  local GPU=$1; shift
  local METHODS=("$@")
  nohup bash -c "
    for M in ${METHODS[*]}; do
      bash $SCRIPTS/timing_clean_v2.sh $GPU \$M
    done
  " > $LOGDIR/timing_clean_g${GPU}_chain.log 2>&1 &
  echo "GPU=$GPU chain=[${METHODS[*]}] pid=$!"
}

echo "[$(date)] launching clean timing on g0, g1, g4, g5"
launch_gpu_chain 0 baseline safedenoiser sld_strong
launch_gpu_chain 1 safree sld_weak
launch_gpu_chain 4 sgf sld_medium
launch_gpu_chain 5 sld_max ebsg

echo
echo "Monitor progress:"
echo "  ssh siml-05 'tail -F $LOGDIR/timing_clean_*_chain.log'"
echo
echo "After all done (~35 min), merge CSVs by running:"
echo "  bash $SCRIPTS/merge_timing_clean.sh"
