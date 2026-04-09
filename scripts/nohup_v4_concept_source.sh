#!/bin/bash
# =============================================================================
# v4 Concept Source Comparison: text vs img_16 vs img_32
# CAS threshold=0.6, anchor_inpaint, Ring-A-Bell 78 prompts × 4 samples
# =============================================================================
set -e

cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG

PROMPTS="prompts/ringabell.txt"
NSAMPLES=4
CAS=0.6
ST=0.1
SS=1.0
MODE="anchor_inpaint"
STEPS=50

OUTBASE="outputs/v4_concept_source"

# Config 1: text (baseline) — GPU 0
CUDA_VISIBLE_DEVICES=0 nohup conda run -n sdd_copy python3 generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/text_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source text \
    > "${OUTBASE}/text_cas06.log" 2>&1 &

# Config 2: img_16 (CLIP image 16 exemplars) — GPU 3
CUDA_VISIBLE_DEVICES=3 nohup conda run -n sdd_copy python3 generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/img16_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source img_16 \
    > "${OUTBASE}/img16_cas06.log" 2>&1 &

# Config 3: img_32 (CLIP image 32 exemplars) — GPU 4
CUDA_VISIBLE_DEVICES=4 nohup conda run -n sdd_copy python3 generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/img32_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source img_32 \
    > "${OUTBASE}/img32_cas06.log" 2>&1 &

echo "Launched 3 experiments on GPUs 0, 3, 4"
echo "Output: ${OUTBASE}/{text_cas06, img16_cas06, img32_cas06}"
echo "Monitor: tail -f ${OUTBASE}/*.log"

wait
echo "All experiments completed!"
