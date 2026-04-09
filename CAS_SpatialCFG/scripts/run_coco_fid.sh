#!/bin/bash
# COCO 1000장 생성 + FID/CLIP 평가
# GPU 6: baseline (no guidance)
# GPU 7: best CAS 0.6 (V4 ainp s=1.0 sthr=0.1)

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
OUTBASE="$BASE/outputs/coco_fid"
PY4="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/generate_v4.py"
EVALPY="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python $BASE/eval_fid_clip.py"
PROMPTS="$BASE/prompts/coco_250.txt"
LOG="$OUTBASE/run.log"
mkdir -p "$OUTBASE"

COMMON="--nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5"

echo "[$(date)] Starting COCO FID/CLIP generation" | tee -a "$LOG"

# GPU 6: Baseline (safety_scale=0, no guidance)
baseline_worker() {
    echo "[$(date)] GPU6: baseline generation start" >> "$LOG"
    CUDA_VISIBLE_DEVICES=6 $PY4 \
        --prompts $PROMPTS \
        --outdir $OUTBASE/baseline \
        --safety_scale 0.0 \
        --cas_threshold 99.0 \
        --guide_mode anchor_inpaint \
        --spatial_threshold 0.3 \
        $COMMON >> "$LOG" 2>&1
    echo "[$(date)] GPU6: baseline done ($(ls $OUTBASE/baseline/*.png 2>/dev/null | wc -l) images)" >> "$LOG"
}

# GPU 7: Best CAS 0.6 - V4 anchor_inpaint s=1.0 sthr=0.1
method_worker() {
    echo "[$(date)] GPU7: CAS0.6 s1.0 sthr0.1 generation start" >> "$LOG"
    CUDA_VISIBLE_DEVICES=7 $PY4 \
        --prompts $PROMPTS \
        --outdir $OUTBASE/v4_ainp_s10_t01_cas06 \
        --cas_threshold 0.6 \
        --safety_scale 1.0 \
        --spatial_threshold 0.1 \
        --guide_mode anchor_inpaint \
        --cas_sticky \
        $COMMON >> "$LOG" 2>&1
    echo "[$(date)] GPU7: method done ($(ls $OUTBASE/v4_ainp_s10_t01_cas06/*.png 2>/dev/null | wc -l) images)" >> "$LOG"
}

# Run both in parallel
baseline_worker &
PID_BASE=$!
method_worker &
PID_METH=$!

wait $PID_BASE $PID_METH
echo "[$(date)] Both generation done" | tee -a "$LOG"

# FID + CLIP eval on GPU 6
echo "[$(date)] Computing FID + CLIP..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=6 $EVALPY \
    $OUTBASE/baseline \
    $OUTBASE/v4_ainp_s10_t01_cas06 \
    $PROMPTS >> "$LOG" 2>&1

echo "[$(date)] ALL DONE" | tee -a "$LOG"
cat $OUTBASE/v4_ainp_s10_t01_cas06/results_fid_clip.txt 2>/dev/null
