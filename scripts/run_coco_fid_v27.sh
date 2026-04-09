#!/bin/bash
# COCO FID + CLIP Score for v27 best configs
# Requires: baseline already generated in coco_fid/baseline (1000 imgs, nsamples=4)

set -euo pipefail

GPU=${1:-6}
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
EVALPY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/eval_fid_clip.py"
CLIP=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
PROMPTS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/coco_250.txt
OUTBASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/coco_fid
BASELINE=$OUTBASE/baseline

COMMON="--nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 --cas_threshold 0.6"
BOTH="--probe_mode both --clip_embeddings $CLIP --attn_threshold 0.1 --img_attn_threshold 0.4"
TXT="--probe_mode text --attn_threshold 0.1"

echo "[$(date)] Starting COCO FID/CLIP for v27 best configs on GPU $GPU"

# Check baseline exists
N_BASE=$(find $BASELINE -name "*.png" 2>/dev/null | wc -l)
echo "Baseline images: $N_BASE"
if [ $N_BASE -lt 500 ]; then
    echo "ERROR: Baseline needs more images. Generating..."
    CUDA_VISIBLE_DEVICES=$GPU $P $V27 \
        --prompts $PROMPTS --outdir $BASELINE \
        --nsamples 4 --steps 50 --seed 42 --cfg_scale 7.5 \
        --cas_threshold 99.0 --safety_scale 0.0 \
        --how_mode anchor_inpaint \
        --probe_mode text --attn_threshold 0.1
fi

generate_and_eval() {
    local name=$1; shift
    local outdir=$OUTBASE/$name

    if [ -f "${outdir}/results_fid_clip.txt" ]; then
        echo "[SKIP] Already evaluated: $name"
        cat "${outdir}/results_fid_clip.txt"
        return
    fi

    # Generate if needed
    local nimgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    if [ $nimgs -lt 500 ]; then
        echo "[$(date)] Generating $name ($nimgs existing)..."
        CUDA_VISIBLE_DEVICES=$GPU $P $V27 \
            --prompts $PROMPTS --outdir "$outdir" $COMMON "$@"
    fi

    nimgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    echo "[$(date)] Evaluating $name ($nimgs imgs)..."

    CUDA_VISIBLE_DEVICES=$GPU $EVALPY "$BASELINE" "$outdir" "$PROMPTS"
    echo ""
    cat "${outdir}/results_fid_clip.txt" 2>/dev/null
    echo ""
}

# v27 hyb both (best overall: ts=15 as=15)
generate_and_eval "v27_hyb_ts15_as15_both" \
    --how_mode hybrid --target_scale 15 --anchor_scale 15 $BOTH

# v27 ainp both (ss=1.2)
generate_and_eval "v27_ainp_ss12_both" \
    --how_mode anchor_inpaint --safety_scale 1.2 $BOTH

# v27 ainp text-only (ss=1.2)
generate_and_eval "v27_ainp_ss12_text" \
    --how_mode anchor_inpaint --safety_scale 1.2 $TXT

# v27 hyb text-only (ts=15 as=15)
generate_and_eval "v27_hyb_ts15_as15_text" \
    --how_mode hybrid --target_scale 15 --anchor_scale 15 $TXT

echo ""
echo "=============================================="
echo "  COCO FID/CLIP DONE — $(date)"
echo "=============================================="
echo ""
echo "=== Summary of ALL FID results ==="
for d in $OUTBASE/*/; do
    if [ -f "${d}results_fid_clip.txt" ]; then
        name=$(basename "$d")
        fid=$(grep "FID" "${d}results_fid_clip.txt" | grep -oP '[\d.]+$')
        clip_base=$(grep "CLIP Score (Baseline)" "${d}results_fid_clip.txt" | grep -oP '[\d.]+$')
        clip_meth=$(grep "CLIP Score (Method)" "${d}results_fid_clip.txt" | grep -oP '[\d.]+$')
        printf "%-40s FID=%-8s CLIP_base=%-8s CLIP_meth=%-8s\n" "$name" "$fid" "$clip_base" "$clip_meth"
    fi
done
