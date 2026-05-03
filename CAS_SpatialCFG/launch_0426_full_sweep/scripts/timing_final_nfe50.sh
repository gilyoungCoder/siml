#!/usr/bin/env bash
# Final headline NFE=50 timing. One method per GPU, no contention.
# Writes: paper_results/figures/nfe50_final_<METHOD>.csv (1 row).
# Usage: bash timing_final_nfe50.sh <GPU> <METHOD>
set -uo pipefail
GPU=${1:?gpu}
METHOD=${2:?method}
N=20
NFE=50
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$ROOT/outputs/phase_nfe50_final
RESULT_CSV=$ROOT/paper_results/figures/nfe50_final_${METHOD}.csv
LOG=$ROOT/logs/nfe50_final_${METHOD}_g${GPU}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE $ROOT/logs $ROOT/paper_results/figures

echo "method,nfe,n_imgs,wall_sec_total,per_img_sec_with_load,per_img_sec_excl_load_mtime" > $RESULT_CSV
echo "[$(date)] start GPU=$GPU METHOD=$METHOD NFE=$NFE N=$N" > $LOG

TXT=/tmp/violence_${N}_final_${METHOD}_$$.txt
SRCCSV=$ROOT/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430/prompts/i2p_q16_csv/violence_q16_top60.csv
tail -n +2 "$SRCCSV" | head -n $N | awk -F',' '{print $1}' > $TXT

OUT=$OUTBASE/${METHOD}_g${GPU}
rm -rf $OUT && mkdir -p $OUT

run_sld () {
  local VARIANT=$1
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
    $ROOT/scripts/sld_runner.py --prompts "$TXT" --outdir "$OUT" \
    --variant $VARIANT --steps $NFE --seed 42 --cfg_scale 7.5 >> $LOG 2>&1 || true
}

run_ebsg () {
  cd /mnt/home3/yhgil99/unlearning/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 -m safegen.generate_family \
    --prompts "$TXT" --outdir "$OUT" \
    --family_guidance --family_config /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/i2p_v1/violence/clip_grouped.pt \
    --probe_mode both --probe_fusion union --how_mode hybrid \
    --cas_threshold 0.4 --safety_scale 20.0 \
    --attn_threshold 0.30 --img_attn_threshold 0.10 \
    --n_img_tokens 4 --steps $NFE --seed 42 --cfg_scale 7.5 \
    --target_concepts violence bloody_scene weapon fight \
    --target_words violence bloody scene weapon fight >> $LOG 2>&1 || true
  cd - > /dev/null
}

START=$(date +%s.%N)
case $METHOD in
  sld_max)    run_sld Max ;;
  sld_strong) run_sld Strong ;;
  sld_medium) run_sld Medium ;;
  sld_weak)   run_sld Weak ;;
  ebsg)       run_ebsg ;;
  *) echo "unknown method: $METHOD"; exit 1 ;;
esac
END=$(date +%s.%N)

# Measure (mtime-sorted, fixes the alphabetical bug)
files=$(ls $OUT/*.png 2>/dev/null)
n=$(echo "$files" | sed '/^$/d' | wc -l)
wall=$(echo "$START $END" | awk '{printf "%.2f", $2-$1}')
pi_load="NA"; pi_excl="NA"
[ "$n" -gt 0 ] && pi_load=$(echo "$wall $n" | awk '{printf "%.4f", $1/$2}')
if [ "$n" -gt 1 ]; then
  first=$(echo "$files" | xargs -d '\n' stat -c %Y 2>/dev/null | sort -n | head -1)
  last=$(echo "$files" | xargs -d '\n' stat -c %Y 2>/dev/null | sort -n | tail -1)
  pi_excl=$(echo "$first $last $n" | awk '{printf "%.4f", ($2-$1)/($3-1)}')
fi
echo "${METHOD},${NFE},${n},${wall},${pi_load},${pi_excl}" >> $RESULT_CSV
echo "[$(date)] DONE METHOD=$METHOD n=$n wall=${wall}s per_img(excl)=${pi_excl}s" | tee -a $LOG
rm -f $TXT
