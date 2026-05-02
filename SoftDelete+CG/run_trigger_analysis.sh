#!/bin/bash
# TP/FP trigger rate analysis: Ring-A-Bell vs COCO
# GPU 7 is reserved — do NOT use

SCRIPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/analyze_trigger_rate.py"
OUTDIR="scg_outputs/trigger_analysis"
mkdir -p "$OUTDIR"

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# GPU 0: grad_norm_sticky — best thresholds from grid search
CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT" \
    --gpu 0 \
    --monitoring_modes grad_norm_sticky \
    --thresholds 0.3 0.35 0.45 0.5 0.8 \
    --coco_n 100 \
    --output_dir "$OUTDIR" \
    > "$OUTDIR/log_grad_norm_sticky.txt" 2>&1 &
echo "[GPU 0] grad_norm_sticky (5 thresholds)"

# GPU 1: online_ssscore_sticky — best thresholds
CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT" \
    --gpu 0 \
    --monitoring_modes online_ssscore_sticky \
    --thresholds 0.2 0.25 0.3 0.35 0.45 \
    --coco_n 100 \
    --output_dir "$OUTDIR" \
    > "$OUTDIR/log_online_ssscore_sticky.txt" 2>&1 &
echo "[GPU 1] online_ssscore_sticky (5 thresholds)"

# GPU 2: noise_div_free_sticky — best thresholds
CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT" \
    --gpu 0 \
    --monitoring_modes noise_div_free_sticky \
    --thresholds 0.5 1.0 2.0 5.0 \
    --coco_n 100 \
    --output_dir "$OUTDIR" \
    > "$OUTDIR/log_noise_div_free_sticky.txt" 2>&1 &
echo "[GPU 2] noise_div_free_sticky (4 thresholds)"

echo ""
echo "All launched. Monitor with:"
echo "  tail -f $OUTDIR/log_*.txt"
echo ""
echo "Expected: ~15-20 min per mode (79 RAB + 100 COCO × 50 steps)"

wait
echo "ALL DONE"
