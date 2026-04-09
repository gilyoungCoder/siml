#!/bin/bash
BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
OUTBASE="$BASE/outputs/top15_sweep"
PY4="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v4.py"
PY5="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v5.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COMMON="--nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 --cas_sticky"
LOG="$OUTBASE/gen.log"

gen() { local gpu=$1; local outdir=$2; local script=$3; shift 3
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -ge 316 ] && { echo "[SKIP] $(basename $outdir)"; return; }
    echo "[$(date '+%H:%M')] GPU$gpu: $(basename $outdir)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu $script --prompts $PROMPTS --outdir $outdir $COMMON "$@" >> "$LOG" 2>&1
    echo "[$(date '+%H:%M')] GPU$gpu DONE: $(basename $outdir) ($(ls $outdir/*.png 2>/dev/null | wc -l))" >> "$LOG"
}

# 4개 → GPU 1
g1() {
gen 1 $OUTBASE/v4_ainp_s09_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v4_ainp_s09_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v4_ainp_s09_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v5_ainp_s10_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
}

# 3개 → GPU 2
g2() {
gen 2 $OUTBASE/v5_ainp_s10_pm01_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 2 $OUTBASE/v5_ainp_s10_pm01_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 2 $OUTBASE/v5_ainp_s10_pm01_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
}

echo "[$(date)] Rerunning gpu0 jobs on GPU 1,2" >> "$LOG"
g1 & g2 &
wait
echo "[$(date)] gpu0 jobs done" >> "$LOG"
