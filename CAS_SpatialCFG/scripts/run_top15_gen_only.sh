#!/bin/bash
# Top-15 × CAS 0.5/0.6/0.7 — generation ONLY (no eval inside workers)
# GPU 0-5 (GPU 6,7 reserved for COCO FID)
# 42 jobs, 7 per GPU

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
OUTBASE="$BASE/outputs/top15_sweep"
PY4="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v4.py"
PY5="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v5.py"
PY3="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COMMON="--nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 --cas_sticky"
LOG="$OUTBASE/gen.log"
mkdir -p "$OUTBASE"

gen() { local gpu=$1; local outdir=$2; local script=$3; shift 3
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -ge 316 ] && { echo "[SKIP] $outdir already complete"; return; }
    echo "[$(date '+%H:%M')] GPU$gpu GEN START: $(basename $outdir)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu $script --prompts $PROMPTS --outdir $outdir $COMMON "$@" >> "$LOG" 2>&1
    echo "[$(date '+%H:%M')] GPU$gpu GEN DONE: $(basename $outdir) ($(ls $outdir/*.png 2>/dev/null | wc -l) imgs)" >> "$LOG"
}

# ─── GPU 0 (7 jobs) ──────────────────────────────────────────────────────────
gpu0() {
gen 0 $OUTBASE/v4_ainp_s09_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 0 $OUTBASE/v4_ainp_s09_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 0 $OUTBASE/v4_ainp_s09_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 0 $OUTBASE/v5_ainp_s10_pm01_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 0 $OUTBASE/v5_ainp_s10_pm01_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 0 $OUTBASE/v5_ainp_s10_pm01_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 0 $OUTBASE/v5_ainp_s10_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
}

# ─── GPU 1 (7 jobs) ──────────────────────────────────────────────────────────
gpu1() {
gen 1 $OUTBASE/v5_ainp_s10_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 1 $OUTBASE/v5_ainp_s10_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 1 $OUTBASE/v4_ainp_s10_t02_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v4_ainp_s10_t02_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v4_ainp_s10_t02_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.2 --guide_mode anchor_inpaint
gen 1 $OUTBASE/v5_ainp_s09_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 1 $OUTBASE/v5_ainp_s09_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
}

# ─── GPU 2 (7 jobs) ──────────────────────────────────────────────────────────
gpu2() {
gen 2 $OUTBASE/v5_ainp_s09_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 0.9 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 2 $OUTBASE/v4_ainp_s10_t03_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 2 $OUTBASE/v4_ainp_s10_t03_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 2 $OUTBASE/v4_ainp_s10_t03_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 2 $OUTBASE/v4_ainp_s10_t05_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint
gen 2 $OUTBASE/v4_ainp_s10_t05_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint
gen 2 $OUTBASE/v4_ainp_s10_t05_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 1.0 --spatial_threshold 0.5 --guide_mode anchor_inpaint
}

# ─── GPU 3 (7 jobs) ──────────────────────────────────────────────────────────
gpu3() {
gen 3 $OUTBASE/v3_dag_s3_cas05 "$PY3" --cas_threshold 0.5 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v3_dag_s3_cas06 "$PY3" --cas_threshold 0.6 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v3_dag_s3_cas07 "$PY3" --cas_threshold 0.7 --safety_scale 3.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v3_dag_s5_cas05 "$PY3" --cas_threshold 0.5 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v3_dag_s5_cas06 "$PY3" --cas_threshold 0.6 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v3_dag_s5_cas07 "$PY3" --cas_threshold 0.7 --safety_scale 5.0 --spatial_threshold 0.3 --guide_mode dag_adaptive
gen 3 $OUTBASE/v4_ainp_s08_cas05 "$PY4" --cas_threshold 0.5 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint
}

# ─── GPU 4 (7 jobs) ──────────────────────────────────────────────────────────
gpu4() {
gen 4 $OUTBASE/v4_ainp_s08_cas06 "$PY4" --cas_threshold 0.6 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 4 $OUTBASE/v4_ainp_s08_cas07 "$PY4" --cas_threshold 0.7 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint
gen 4 $OUTBASE/v5_ainp_pt_m02_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2
gen 4 $OUTBASE/v5_ainp_pt_m02_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2
gen 4 $OUTBASE/v5_ainp_pt_m02_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.2
gen 4 $OUTBASE/v5_sld_s10_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld
gen 4 $OUTBASE/v5_sld_s10_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld
}

# ─── GPU 5 (7 jobs) ──────────────────────────────────────────────────────────
gpu5() {
gen 5 $OUTBASE/v5_sld_s10_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 10.0 --spatial_threshold 0.3 --guide_mode sld
gen 5 $OUTBASE/v5_ainp_s08_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 5 $OUTBASE/v5_ainp_s08_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 5 $OUTBASE/v5_ainp_s08_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 0.8 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold 0.0
gen 5 $OUTBASE/v5_ainp_pt_m01_cas05 "$PY5" --cas_threshold 0.5 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 5 $OUTBASE/v5_ainp_pt_m01_cas06 "$PY5" --cas_threshold 0.6 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
gen 5 $OUTBASE/v5_ainp_pt_m01_cas07 "$PY5" --cas_threshold 0.7 --safety_scale 0.7 --spatial_threshold 0.3 --guide_mode anchor_inpaint --proj_threshold -0.1
}

echo "[$(date)] Starting top15 generation-only (GPU 0-5, 42 jobs)" | tee -a "$LOG"
gpu0 & gpu1 & gpu2 & gpu3 & gpu4 & gpu5 &
wait
echo "[$(date)] ALL GENERATION DONE" | tee -a "$LOG"
