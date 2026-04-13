#!/bin/bash
# Master pipeline: Phase 0 fix -> Phase 1 generation -> Phase 2 evaluation
# Run with: nohup bash scripts/master_pipeline.sh > scripts/master_pipeline.log 2>&1 &
set -e
cd /mnt/home3/yhgil99/unlearning

echo "=========================================="
echo "  MASTER PIPELINE START: $(date)"
echo "=========================================="

# Phase 0: Fix CLIP + generate missing exemplars
echo ""
echo "[$(date)] Phase 0: Fixing CLIP extraction..."
bash scripts/phase0_fix_clip.sh
echo "[$(date)] Phase 0 complete"

# Verify all .pt files exist
echo ""
echo "Verifying exemplar .pt files..."
for concept in sexual violent disturbing illegal harassment hate selfharm; do
    pt="CAS_SpatialCFG/exemplars/concepts_v2/${concept}/clip_grouped.pt"
    if [ ! -f "$pt" ]; then
        echo "ERROR: Missing $pt"
        exit 1
    fi
    echo "  OK: $pt"
done

# Phase 1: Generate Ours images
echo ""
echo "[$(date)] Phase 1: Generating SafeGen images..."
bash scripts/phase1_ours_generation.sh
echo "[$(date)] Phase 1 complete"

# Phase 2: Evaluate everything
echo ""
echo "[$(date)] Phase 2: Evaluating all outputs..."
bash scripts/phase2_evaluate_all.sh
echo "[$(date)] Phase 2 complete"

echo ""
echo "=========================================="
echo "  MASTER PIPELINE COMPLETE: $(date)"
echo "=========================================="
