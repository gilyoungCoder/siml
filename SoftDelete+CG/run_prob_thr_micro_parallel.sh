#!/bin/bash
# ============================================================================
# Compare MICRO prob_thr values in parallel (0.0001, 0.0005, 0.001, 0.005)
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

BASE_OUTPUT="./scg_outputs/guidance_comparison_prob_thr_micro"
NUM_IMAGES=9

mkdir -p "${BASE_OUTPUT}"

echo "=============================================="
echo "MICRO PROB THRESHOLD COMPARISON - PARALLEL"
echo "=============================================="

# GPU 0: prob_thr=0.0001
CUDA_VISIBLE_DEVICES=0 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.0001 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.0001" &
echo "[GPU 0] prob_thr=0.0001 started"

# GPU 1: prob_thr=0.0005
CUDA_VISIBLE_DEVICES=1 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.0005 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.0005" &
echo "[GPU 1] prob_thr=0.0005 started"

# GPU 2: prob_thr=0.001
CUDA_VISIBLE_DEVICES=2 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.001 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.001" &
echo "[GPU 2] prob_thr=0.001 started"

# GPU 3: prob_thr=0.005
CUDA_VISIBLE_DEVICES=3 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --prob_thr 0.005 \
    --output_dir "${BASE_OUTPUT}/prob_thr_0.005" &
echo "[GPU 3] prob_thr=0.005 started"

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
