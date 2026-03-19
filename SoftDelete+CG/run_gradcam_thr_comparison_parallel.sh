#!/bin/bash
# ============================================================================
# Compare different gradcam_thr values in parallel (CDF-normalized threshold)
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

BASE_OUTPUT="./scg_outputs/guidance_comparison_gradcam_thr"
NUM_IMAGES=9

mkdir -p "${BASE_OUTPUT}"

echo "=============================================="
echo "GRADCAM CDF THRESHOLD COMPARISON - PARALLEL"
echo "=============================================="

# GPU 0: gradcam_thr=0.1
CUDA_VISIBLE_DEVICES=0 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.1 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.1" &
echo "[GPU 0] gradcam_thr=0.1 started"

# GPU 1: gradcam_thr=0.2
CUDA_VISIBLE_DEVICES=1 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.2 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.2" &
echo "[GPU 1] gradcam_thr=0.2 started"

# GPU 2: gradcam_thr=0.3
CUDA_VISIBLE_DEVICES=2 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.3 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.3" &
echo "[GPU 2] gradcam_thr=0.3 started"

# GPU 3: gradcam_thr=0.4
CUDA_VISIBLE_DEVICES=3 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.4 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.4" &
echo "[GPU 3] gradcam_thr=0.4 started"

# GPU 4: gradcam_thr=0.5
CUDA_VISIBLE_DEVICES=4 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.5 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.5" &
echo "[GPU 4] gradcam_thr=0.5 started"

# GPU 5: gradcam_thr=0.6
CUDA_VISIBLE_DEVICES=5 python compare_guidance_methods.py \
    --num_images ${NUM_IMAGES} \
    --gradcam_thr 0.6 \
    --output_dir "${BASE_OUTPUT}/gradcam_thr_0.6" &
echo "[GPU 5] gradcam_thr=0.6 started"

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
