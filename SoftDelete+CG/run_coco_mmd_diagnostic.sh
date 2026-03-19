#!/bin/bash
# COCO MMD diagnostic: get MMD values for benign prompts
# to compare with Ring-A-Bell harmful prompt MMD values
# GPU 7 is reserved - DO NOT USE
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

COCO_PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k_10k.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_DIR="scg_outputs/mmd_gated_v2/coco_diagnostic"

# Use first 50 COCO prompts only
END_IDX=50

# Use threshold=0 so everything triggers (we just want the MMD values for analysis)
CUDA_VISIBLE_DEVICES=0 python generate_mmd_gated_ascg.py \
    --ckpt_path CompVis/stable-diffusion-v1-4 \
    --prompt_file "${COCO_PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
    --sgf_ref_path "${SGF_REF_PATH}" \
    --mmd_metric "kernel_sim" \
    --mmd_threshold 0.0 \
    --mmd_window_start 1000 \
    --mmd_window_end 400 \
    --ascg_scale 0.0 \
    --ascg_base_scale 0.0 \
    --spatial_threshold_start 0.2 \
    --spatial_threshold_end 0.3 \
    --spatial_threshold_strategy cosine \
    --guidance_schedule linear \
    --end_idx 50

echo "COCO diagnostic done!"

# Analysis
python3 << 'PYEOF'
import json, numpy as np

# Load COCO results
coco = json.load(open("scg_outputs/mmd_gated_v2/coco_diagnostic/generation_stats.json"))
coco_stats = coco['per_image_stats']

# Load Ring-A-Bell results (from v2 grid search)
rab = json.load(open("scg_outputs/mmd_gated_v2/ksim_t0.024_sticky/generation_stats.json"))
rab_stats = rab['per_image_stats']

print("\n" + "="*80)
print("  COCO vs Ring-A-Bell: kernel_sim MMD distribution")
print("="*80)

coco_max = np.array([s['max_mmd'] for s in coco_stats])
coco_avg = np.array([s['avg_mmd'] for s in coco_stats])
rab_max = np.array([s['max_mmd'] for s in rab_stats])
rab_avg = np.array([s['avg_mmd'] for s in rab_stats])

print(f"\n{'Metric':<15s} {'':>5s} {'COCO (benign)':>25s} {'Ring-A-Bell (harmful)':>25s}")
print("-"*75)
print(f"{'max_mmd':<15s} {'min':>5s} {coco_max.min():>25.4f} {rab_max.min():>25.4f}")
print(f"{'':>15s} {'p25':>5s} {np.percentile(coco_max,25):>25.4f} {np.percentile(rab_max,25):>25.4f}")
print(f"{'':>15s} {'med':>5s} {np.median(coco_max):>25.4f} {np.median(rab_max):>25.4f}")
print(f"{'':>15s} {'p75':>5s} {np.percentile(coco_max,75):>25.4f} {np.percentile(rab_max,75):>25.4f}")
print(f"{'':>15s} {'max':>5s} {coco_max.max():>25.4f} {rab_max.max():>25.4f}")

print(f"\n{'avg_mmd':<15s} {'min':>5s} {coco_avg.min():>25.4f} {rab_avg.min():>25.4f}")
print(f"{'':>15s} {'med':>5s} {np.median(coco_avg):>25.4f} {np.median(rab_avg):>25.4f}")
print(f"{'':>15s} {'max':>5s} {coco_avg.max():>25.4f} {rab_avg.max():>25.4f}")

# For each threshold, show TP (RAB triggered) and FP (COCO triggered)
print(f"\n{'Threshold':<12s} {'RAB trig (TP)':>15s} {'COCO trig (FP)':>15s} {'Gap':>10s}")
print("-"*55)
for t in np.arange(0.015, 0.036, 0.001):
    tp = np.sum(rab_max >= t)
    fp = np.sum(coco_max >= t)
    print(f"  {t:.3f}       {tp:>3d}/79 ({100*tp/79:4.0f}%)   {fp:>3d}/50 ({100*fp/50:4.0f}%)   {tp/79 - fp/50:>+.2f}")

# Highlight the sweet spot
print("\n  → Sweet spot: threshold where TP is high and FP is low")
PYEOF
