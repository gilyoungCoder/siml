#!/usr/bin/env bash
# Launch 5 per-method timing scripts in parallel: 1 method per GPU.
# Distribution: g0=baseline, g1=safree, g3=safedenoiser, g4=sgf, g5=ebsg
# After all done, merge per-method CSVs into nfe_walltime_timing.csv
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
SCRIPTS=$ROOT/scripts
LOGDIR=$ROOT/logs
mkdir -p $LOGDIR

declare -A ASSIGN=( [0]=baseline [1]=safree [3]=safedenoiser [4]=sgf [5]=ebsg )

PIDS=()
for GPU in 0 1 3 4 5; do
  METHOD=${ASSIGN[$GPU]}
  LOG=$LOGDIR/parallel_timing_${METHOD}_g${GPU}.log
  nohup bash $SCRIPTS/timing_one_method.sh $GPU $METHOD > $LOG 2>&1 &
  pid=$!
  PIDS+=($pid)
  echo "GPU=$GPU METHOD=$METHOD pid=$pid log=$LOG"
done

echo
echo "[$(date)] all 5 launched. waiting for completion ..."
for pid in "${PIDS[@]}"; do
  wait $pid 2>/dev/null || true
done
echo "[$(date)] all 5 done. merging CSVs ..."

OUT=$ROOT/paper_results/figures/nfe_walltime_timing.csv
echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $OUT
for METHOD in baseline safree safedenoiser sgf ebsg; do
  PCSV=$ROOT/paper_results/figures/nfe_walltime_timing_${METHOD}.csv
  [ -f "$PCSV" ] && tail -n +2 "$PCSV" >> $OUT
done
echo "merged → $OUT ($(wc -l < $OUT) lines)"
echo "[$(date)] done."
