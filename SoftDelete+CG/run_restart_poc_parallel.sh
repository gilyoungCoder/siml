#!/bin/bash
# =============================================================================
# ASCG + Guided Restart Sampling - Parallel across 8 GPUs
#
# 8 experiments, 1 per GPU:
#   GPU 0: baseline (no restart)
#   GPU 1: restart_t=100, gf=0.0
#   GPU 2: restart_t=200, gf=0.0
#   GPU 3: restart_t=300, gf=0.0
#   GPU 4: restart_t=400, gf=0.0
#   GPU 5: restart_t=200, gf=0.2
#   GPU 6: restart_t=300, gf=0.2
#   GPU 7: restart_t=200, gf=0.0, count=2
# =============================================================================

set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ===== Common Config =====
MODEL="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLF="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json"
BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/restart_poc"
LOGDIR="${BASE}/logs"
mkdir -p "$LOGDIR"

# ASCG settings
GS=10.0; SP_S=0.3; SP_E=0.5; HS=1.0; BGS=0.0

COMMON_ARGS="$MODEL \
    --prompt_file $PROMPT_FILE \
    --classifier_ckpt $CLF \
    --gradcam_stats_file $STATS \
    --num_classes 4 --harmful_class 2 --safe_class 1 \
    --guidance_scale $GS \
    --spatial_threshold_start $SP_S --spatial_threshold_end $SP_E \
    --threshold_strategy cosine_anneal \
    --use_bidirectional --harmful_scale $HS --base_guidance_scale $BGS \
    --seed 42 --nsamples 1 --num_inference_steps 50 --cfg_scale 7.5 \
    --save_comparison --safety_check"

echo "=============================================="
echo "Launching 8 experiments on 8 GPUs..."
echo "=============================================="

# GPU 0: Baseline (no restart)
echo "[GPU 0] baseline_no_restart"
CUDA_VISIBLE_DEVICES=0 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 0 --restart_count 0 \
    --output_dir "${BASE}/baseline_no_restart" \
    > "${LOGDIR}/baseline_no_restart.log" 2>&1 &

# GPU 1: restart_t=100
echo "[GPU 1] restart_t100_gf0.0_rc1"
CUDA_VISIBLE_DEVICES=1 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 100 --restart_guidance_fraction 0.0 --restart_count 1 \
    --output_dir "${BASE}/restart_t100_gf0.0_rc1" \
    > "${LOGDIR}/restart_t100_gf0.0_rc1.log" 2>&1 &

# GPU 2: restart_t=200
echo "[GPU 2] restart_t200_gf0.0_rc1"
CUDA_VISIBLE_DEVICES=2 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 200 --restart_guidance_fraction 0.0 --restart_count 1 \
    --output_dir "${BASE}/restart_t200_gf0.0_rc1" \
    > "${LOGDIR}/restart_t200_gf0.0_rc1.log" 2>&1 &

# GPU 3: restart_t=300
echo "[GPU 3] restart_t300_gf0.0_rc1"
CUDA_VISIBLE_DEVICES=3 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 300 --restart_guidance_fraction 0.0 --restart_count 1 \
    --output_dir "${BASE}/restart_t300_gf0.0_rc1" \
    > "${LOGDIR}/restart_t300_gf0.0_rc1.log" 2>&1 &

# GPU 4: restart_t=400
echo "[GPU 4] restart_t400_gf0.0_rc1"
CUDA_VISIBLE_DEVICES=4 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 400 --restart_guidance_fraction 0.0 --restart_count 1 \
    --output_dir "${BASE}/restart_t400_gf0.0_rc1" \
    > "${LOGDIR}/restart_t400_gf0.0_rc1.log" 2>&1 &

# GPU 5: restart_t=200 with partial guidance
echo "[GPU 5] restart_t200_gf0.2_rc1"
CUDA_VISIBLE_DEVICES=5 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 200 --restart_guidance_fraction 0.2 --restart_count 1 \
    --output_dir "${BASE}/restart_t200_gf0.2_rc1" \
    > "${LOGDIR}/restart_t200_gf0.2_rc1.log" 2>&1 &

# GPU 6: restart_t=300 with partial guidance
echo "[GPU 6] restart_t300_gf0.2_rc1"
CUDA_VISIBLE_DEVICES=6 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 300 --restart_guidance_fraction 0.2 --restart_count 1 \
    --output_dir "${BASE}/restart_t300_gf0.2_rc1" \
    > "${LOGDIR}/restart_t300_gf0.2_rc1.log" 2>&1 &

# GPU 7: restart_t=200, 2 cycles
echo "[GPU 7] restart_t200_gf0.0_rc2"
CUDA_VISIBLE_DEVICES=7 python generate_ascg_restart.py $COMMON_ARGS \
    --restart_timestep 200 --restart_guidance_fraction 0.0 --restart_count 2 \
    --output_dir "${BASE}/restart_t200_gf0.0_rc2" \
    > "${LOGDIR}/restart_t200_gf0.0_rc2.log" 2>&1 &

echo ""
echo "All 8 experiments launched! Waiting for completion..."
echo "Logs: ${LOGDIR}/"
echo ""

wait

echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""

# Print summary
for dir in "${BASE}"/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/generation_stats.json" ]; then
        total=$(python3 -c "import json; d=json.load(open('$dir/generation_stats.json')); print(d['summary']['total'])" 2>/dev/null || echo "?")
        fallback=$(python3 -c "import json; d=json.load(open('$dir/generation_stats.json')); print(d['summary']['safety_fallback'])" 2>/dev/null || echo "?")
        echo "  $name: ${total} images, ${fallback} safety fallbacks"
    fi
done

echo ""
echo "Next: Run VLM evaluation on each variant"
