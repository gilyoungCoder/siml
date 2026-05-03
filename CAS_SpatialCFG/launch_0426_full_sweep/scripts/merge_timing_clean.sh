#!/usr/bin/env bash
# Merge per-method clean timing CSVs into nfe_walltime_timing.csv (and SLD CSV).
# Backs up old CSVs.
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUT=$ROOT/paper_results/figures/nfe_walltime_timing.csv
SLD_OUT=$ROOT/paper_results/figures/nfe_walltime_timing_sld.csv

cp $OUT ${OUT}.bak_$(date +%m%d_%H%M) 2>/dev/null || true
cp $SLD_OUT ${SLD_OUT}.bak_$(date +%m%d_%H%M) 2>/dev/null || true

echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $OUT
for METHOD in baseline safree safedenoiser sgf sld_max sld_medium sld_strong sld_weak ebsg; do
  PCSV=$ROOT/paper_results/figures/nfe_walltime_timing_clean_${METHOD}.csv
  if [ -f "$PCSV" ]; then
    tail -n +2 "$PCSV" >> $OUT
  else
    echo "MISSING: $PCSV"
  fi
done
echo "merged main → $OUT ($(wc -l < $OUT) lines)"

# Refresh SLD-only CSV with the cleaned numbers
echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $SLD_OUT
for METHOD in sld_max sld_medium sld_strong sld_weak; do
  PCSV=$ROOT/paper_results/figures/nfe_walltime_timing_clean_${METHOD}.csv
  [ -f "$PCSV" ] && tail -n +2 "$PCSV" >> $SLD_OUT
done
echo "merged sld → $SLD_OUT ($(wc -l < $SLD_OUT) lines)"

# Re-run the polished plot with new timing data
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
echo "regenerating plot ..."
$PY $ROOT/scripts/nfe_walltime_pareto_polished.py 2>&1 | tail -5
echo "DONE."
