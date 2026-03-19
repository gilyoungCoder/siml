#!/bin/bash
# Generate COCO 500 images for FID evaluation
# Compare: baseline (no guidance) vs online_ssscore_sticky at various thresholds
# GPU 0: busy with trigger analysis
# GPU 7: reserved (other user)
# Use GPUs 1-6

set -e
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

GEN_SCRIPT="generate_unified_monitoring.py"
COCO_CSV="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"
CLASSIFIER="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell"
OUTBASE="scg_outputs/coco_fid"
mkdir -p "$OUTBASE"

N=500  # generate first 500 COCO prompts

# ──────────────────────────────────────────────────
# GPU 1: Baseline — no guidance (just vanilla SD)
# We use a very high threshold so nothing triggers
# ──────────────────────────────────────────────────
echo "[GPU 1] Baseline (no guidance)"
CUDA_VISIBLE_DEVICES=1 python3 "$GEN_SCRIPT" \
    --ckpt_path runwayml/stable-diffusion-v1-5 \
    --prompt_file "$COCO_CSV" \
    --classifier_ckpt "$CLASSIFIER" \
    --gradcam_stats_dir "$GRADCAM_STATS" \
    --monitoring_mode online_ssscore_sticky \
    --monitoring_threshold 999.0 \
    --ssscore_target nudity \
    --spatial_threshold_start 0.2 --spatial_threshold_end 0.4 \
    --guidance_scale 10.0 --base_guidance_scale 3.0 \
    --num_inference_steps 50 --nsamples 1 \
    --start_idx 0 --end_idx $N \
    --output_dir "$OUTBASE/baseline" \
    > "$OUTBASE/log_baseline.txt" 2>&1 &

# ──────────────────────────────────────────────────
# GPU 2: online_ssscore_sticky thr=0.3 (SR=78.5%)
# ──────────────────────────────────────────────────
echo "[GPU 2] online_ssscore_sticky thr=0.3"
CUDA_VISIBLE_DEVICES=2 python3 "$GEN_SCRIPT" \
    --ckpt_path runwayml/stable-diffusion-v1-5 \
    --prompt_file "$COCO_CSV" \
    --classifier_ckpt "$CLASSIFIER" \
    --gradcam_stats_dir "$GRADCAM_STATS" \
    --monitoring_mode online_ssscore_sticky \
    --monitoring_threshold 0.3 \
    --ssscore_target nudity \
    --spatial_threshold_start 0.2 --spatial_threshold_end 0.4 \
    --guidance_scale 10.0 --base_guidance_scale 3.0 \
    --num_inference_steps 50 --nsamples 1 \
    --start_idx 0 --end_idx $N \
    --output_dir "$OUTBASE/online_ssscore_sticky_thr0.3" \
    > "$OUTBASE/log_thr0.3.txt" 2>&1 &

# ──────────────────────────────────────────────────
# GPU 3: online_ssscore_sticky thr=0.45
# ──────────────────────────────────────────────────
echo "[GPU 3] online_ssscore_sticky thr=0.45"
CUDA_VISIBLE_DEVICES=3 python3 "$GEN_SCRIPT" \
    --ckpt_path runwayml/stable-diffusion-v1-5 \
    --prompt_file "$COCO_CSV" \
    --classifier_ckpt "$CLASSIFIER" \
    --gradcam_stats_dir "$GRADCAM_STATS" \
    --monitoring_mode online_ssscore_sticky \
    --monitoring_threshold 0.45 \
    --ssscore_target nudity \
    --spatial_threshold_start 0.2 --spatial_threshold_end 0.4 \
    --guidance_scale 10.0 --base_guidance_scale 3.0 \
    --num_inference_steps 50 --nsamples 1 \
    --start_idx 0 --end_idx $N \
    --output_dir "$OUTBASE/online_ssscore_sticky_thr0.45" \
    > "$OUTBASE/log_thr0.45.txt" 2>&1 &

# ──────────────────────────────────────────────────
# GPU 4: online_ssscore_sticky thr=0.6
# ──────────────────────────────────────────────────
echo "[GPU 4] online_ssscore_sticky thr=0.6"
CUDA_VISIBLE_DEVICES=4 python3 "$GEN_SCRIPT" \
    --ckpt_path runwayml/stable-diffusion-v1-5 \
    --prompt_file "$COCO_CSV" \
    --classifier_ckpt "$CLASSIFIER" \
    --gradcam_stats_dir "$GRADCAM_STATS" \
    --monitoring_mode online_ssscore_sticky \
    --monitoring_threshold 0.6 \
    --ssscore_target nudity \
    --spatial_threshold_start 0.2 --spatial_threshold_end 0.4 \
    --guidance_scale 10.0 --base_guidance_scale 3.0 \
    --num_inference_steps 50 --nsamples 1 \
    --start_idx 0 --end_idx $N \
    --output_dir "$OUTBASE/online_ssscore_sticky_thr0.6" \
    > "$OUTBASE/log_thr0.6.txt" 2>&1 &

# ──────────────────────────────────────────────────
# GPU 5: grad_norm_sticky thr=0.45 (for comparison — SR best but FP=100%)
# ──────────────────────────────────────────────────
echo "[GPU 5] grad_norm_sticky thr=0.45 (for FID comparison)"
CUDA_VISIBLE_DEVICES=5 python3 "$GEN_SCRIPT" \
    --ckpt_path runwayml/stable-diffusion-v1-5 \
    --prompt_file "$COCO_CSV" \
    --classifier_ckpt "$CLASSIFIER" \
    --gradcam_stats_dir "$GRADCAM_STATS" \
    --monitoring_mode grad_norm_sticky \
    --monitoring_threshold 0.45 \
    --spatial_threshold_start 0.2 --spatial_threshold_end 0.4 \
    --guidance_scale 10.0 --base_guidance_scale 3.0 \
    --num_inference_steps 50 --nsamples 1 \
    --start_idx 0 --end_idx $N \
    --output_dir "$OUTBASE/grad_norm_sticky_thr0.45" \
    > "$OUTBASE/log_grad_norm.txt" 2>&1 &

echo ""
echo "All 5 jobs launched on GPUs 1-5"
echo "Monitor: tail -f $OUTBASE/log_*.txt"
echo "Each generates 500 COCO images → ~25min per job"
echo ""

wait
echo "ALL GENERATION DONE"
echo ""
echo "Computing FID scores..."

# FID computation (baseline as reference)
BASELINE_DIR="$OUTBASE/baseline"

for dir in online_ssscore_sticky_thr0.3 online_ssscore_sticky_thr0.45 online_ssscore_sticky_thr0.6 grad_norm_sticky_thr0.45; do
    echo "FID: $dir vs baseline"
    python3 -m pytorch_fid "$BASELINE_DIR" "$OUTBASE/$dir" --device cuda:1 2>&1 | tee "$OUTBASE/fid_${dir}.txt"
done

echo "ALL DONE"
