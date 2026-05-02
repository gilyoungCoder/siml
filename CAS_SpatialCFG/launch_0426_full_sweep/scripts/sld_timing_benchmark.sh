#!/usr/bin/env bash
# SLD-only isolated timing benchmark on siml-05.
# 4 SLD variants × 8 NFE × 60 violence prompts on a SINGLE GPU (no contention).
# Per-image time computed via PNG mtime range (excludes one-time model load).
# Outputs: paper_results/figures/nfe_walltime_timing_sld.csv
#
# Usage: bash sld_timing_benchmark.sh <GPU_INDEX>
# This script is meant to be invoked AFTER all generation finishes; the wait
# wrapper (`wait_then_sld_timing.sh`) polls GPU idleness before launching this.

set -uo pipefail
GPU=${1:-2}
N=60   # prompts per cell (60-image avg per (variant, NFE))
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_nfe_walltime_timing_sld
LOG=$ROOT/logs/nfe_walltime_timing_sld_g${GPU}_$(date +%m%d_%H%M).log
RESULT_CSV=$ROOT/paper_results/figures/nfe_walltime_timing_sld.csv
mkdir -p $OUTBASE $ROOT/logs $ROOT/paper_results/figures

echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $RESULT_CSV
echo "[$(date)] start GPU=$GPU N=$N (SLD-only timing)" > $LOG

TXT60=/tmp/violence_top${N}_$$.txt
SRCCSV=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv
tail -n +2 "$SRCCSV" | head -n $N | awk -F',' '{print $1}' > $TXT60
echo "[prompts] N=$(wc -l < $TXT60)" | tee -a $LOG

NFES=(5 10 15 20 25 30 40 50)
VARIANTS=(Max Medium Strong Weak)

measure_and_log () {
  local METHOD=$1 NFE=$2 OUTDIR=$3 START=$4 END=$5
  local n=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  local wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
  local pi_load=$(echo "$wall $n" | awk '{if ($2>0) printf "%.4f", $1/$2; else print "NA"}')
  local pi_excl="NA"
  if [ "$n" -gt 1 ]; then
    local first=$(ls -1tr $OUTDIR/*.png | head -1)
    local last=$(ls -1tr $OUTDIR/*.png | tail -1)
    local first_t=$(stat -c %Y "$first")
    local last_t=$(stat -c %Y "$last")
    pi_excl=$(echo "$first_t $last_t $n" | awk '{if ($3>1) printf "%.4f", ($2-$1)/($3-1); else print "NA"}')
  fi
  echo "${METHOD},${NFE},${n},${wall},${pi_load},${pi_excl}" >> $RESULT_CSV
  echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] n=$n wall=${wall}s per_img(load)=${pi_load}s per_img(excl)=${pi_excl}s" | tee -a $LOG
}

for NFE in "${NFES[@]}"; do
  for VAR in "${VARIANTS[@]}"; do
    METHOD="sld_$(echo $VAR | tr 'A-Z' 'a-z')"
    OUT=$OUTBASE/${METHOD}_violence_nfe${NFE}
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "[$(date +%H:%M:%S)] [$METHOD nfe=$NFE] start" | tee -a $LOG
    START=$(date +%s.%N)
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU \
      /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
      $ROOT/scripts/sld_runner.py \
      --prompts "$TXT60" --outdir "$OUT" \
      --variant $VAR --steps $NFE --seed 42 --cfg_scale 7.5 >> $LOG 2>&1 || true
    END=$(date +%s.%N)
    measure_and_log $METHOD $NFE $OUT $START $END
  done
done

echo "[$(date)] all done. Result CSV: $RESULT_CSV" | tee -a $LOG
echo
echo "=== SUMMARY ==="
cat $RESULT_CSV
rm -f $TXT60
