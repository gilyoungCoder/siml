#!/bin/bash
# ============================================================================
# Compare different prob_thr values in parallel
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

BASE_OUTPUT="./scg_outputs/guidance_comparison_prob_thr"
NUM_IMAGES=9

mkdir -p "${BASE_OUTPUT}"

echo "=============================================="
echo "PROB THRESHOLD COMPARISON - PARALLEL"
echo "=============================================="

# GPU 0: prob_thr=0.01
CUDA_VISIBLE_DEVICES=0 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.01 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.01" &
echo "[GPU 0] prob_thr=0.01 started"

# GPU 1: prob_thr=0.03
CUDA_VISIBLE_DEVICES=1 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.03 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.03" &
echo "[GPU 1] prob_thr=0.03 started"

# GPU 2: prob_thr=0.05
CUDA_VISIBLE_DEVICES=2 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.05 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.05" &
echo "[GPU 2] prob_thr=0.05 started"

# GPU 3: prob_thr=0.1
CUDA_VISIBLE_DEVICES=3 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.1 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.1" &
echo "[GPU 3] prob_thr=0.1 started"

# GPU 4: prob_thr=0.15
CUDA_VISIBLE_DEVICES=4 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.15 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.15" &
echo "[GPU 4] prob_thr=0.15 started"

# GPU 5: prob_thr=0.2
CUDA_VISIBLE_DEVICES=5 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.2 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.2" &
echo "[GPU 5] prob_thr=0.2 started"

echo ""
echo "Waiting for all jobs..."
wait

echo ""
echo "=============================================="
echo "ALL COMPLETE!"
echo "=============================================="
echo "Results: ${BASE_OUTPUT}/"
ls -la "${BASE_OUTPUT}/"
echo "=============================================="
