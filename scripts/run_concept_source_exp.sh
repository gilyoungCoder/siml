#!/bin/bash
# =============================================================================
# v4 Concept Source Comparison: text vs img_16 vs img_32
# CAS threshold=0.6, anchor_inpaint, Ring-A-Bell 78 prompts × 4 samples
# Single GPU sequential run
# =============================================================================
set -e

cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG

PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS="prompts/ringabell.txt"
NSAMPLES=4
CAS=0.6
ST=0.1
SS=1.0
MODE="anchor_inpaint"
STEPS=50

OUTBASE="outputs/v4_concept_source"
mkdir -p "$OUTBASE"

echo "============================================="
echo "Starting v4 concept source comparison"
echo "CAS=$CAS, ST=$ST, SS=$SS, MODE=$MODE"
echo "Ring-A-Bell 78 prompts × $NSAMPLES samples"
echo "============================================="

# Config 1: text (baseline)
echo ""
echo "[1/3] text_cas06 — text labels baseline"
$PYTHON generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/text_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source text
echo "[1/3] DONE"

# Config 2: img_16 (CLIP image 16 exemplars)
echo ""
echo "[2/3] img16_cas06 — CLIP image 16 nudity exemplars"
$PYTHON generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/img16_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source img_16
echo "[2/3] DONE"

# Config 3: img_32 (CLIP image 32 exemplars)
echo ""
echo "[3/3] img32_cas06 — CLIP image 32 nudity exemplars"
$PYTHON generate_v4.py \
    --prompts "$PROMPTS" \
    --outdir "${OUTBASE}/img32_cas06" \
    --nsamples $NSAMPLES --steps $STEPS \
    --cas_threshold $CAS --spatial_threshold $ST \
    --safety_scale $SS --guide_mode $MODE \
    --concept_source img_32
echo "[3/3] DONE"

echo ""
echo "============================================="
echo "All 3 configs completed!"
echo "Results in: ${OUTBASE}/{text_cas06, img16_cas06, img32_cas06}"
echo "============================================="
