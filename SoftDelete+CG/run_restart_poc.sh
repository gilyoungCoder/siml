#!/bin/bash
# =============================================================================
# ASCG + Guided Restart Sampling (GRS) - PoC Experiments
#
# Tests different restart configurations to find the optimal
# safety-quality trade-off.
#
# Experiment grid:
#   - restart_timestep: {100, 200, 300, 400} (noise level for restart)
#   - restart_guidance_fraction: {0.0, 0.2} (how much guidance during restart)
#   - restart_count: {1, 2} (number of restart cycles)
#
# Also generates Phase 1-only baselines for comparison.
# =============================================================================

set -e

# ===== Environment =====
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ===== Configuration =====
MODEL="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json"
BASE_OUTPUT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/restart_poc"

# Phase 1 ASCG settings (use known good configuration)
GUIDANCE_SCALE=10.0
SP_START=0.3
SP_END=0.5
THRESHOLD_STRATEGY="cosine_anneal"
HARMFUL_SCALE=1.0
BASE_GS=0.0
NUM_CLASSES=4
HARMFUL_CLASS=2
SAFE_CLASS=1

# Common
SEED=42
NSAMPLES=1
STEPS=50
CFG=7.5

echo "=============================================="
echo "ASCG + Guided Restart Sampling - PoC Experiments"
echo "=============================================="
echo "Model: $MODEL"
echo "Prompts: $PROMPT_FILE"
echo "Base output: $BASE_OUTPUT"
echo ""

# ===== Experiment 1: Phase 1 Only (Baseline) =====
echo "[Exp 1/8] Phase 1 Only (no restart) - Baseline"
python generate_ascg_restart.py "$MODEL" \
    --prompt_file "$PROMPT_FILE" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --gradcam_stats_file "$GRADCAM_STATS" \
    --num_classes $NUM_CLASSES \
    --harmful_class $HARMFUL_CLASS \
    --safe_class $SAFE_CLASS \
    --guidance_scale $GUIDANCE_SCALE \
    --spatial_threshold_start $SP_START \
    --spatial_threshold_end $SP_END \
    --threshold_strategy $THRESHOLD_STRATEGY \
    --use_bidirectional \
    --harmful_scale $HARMFUL_SCALE \
    --base_guidance_scale $BASE_GS \
    --restart_timestep 0 \
    --restart_count 0 \
    --seed $SEED \
    --nsamples $NSAMPLES \
    --num_inference_steps $STEPS \
    --cfg_scale $CFG \
    --output_dir "${BASE_OUTPUT}/baseline_no_restart" \
    --save_comparison

# ===== Experiment 2-5: Varying restart_timestep =====
for RT in 100 200 300 400; do
    EXP_NAME="restart_t${RT}_gf0.0_rc1"
    echo ""
    echo "[Exp] restart_timestep=${RT}, guidance_fraction=0.0, count=1"

    python generate_ascg_restart.py "$MODEL" \
        --prompt_file "$PROMPT_FILE" \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --gradcam_stats_file "$GRADCAM_STATS" \
        --num_classes $NUM_CLASSES \
        --harmful_class $HARMFUL_CLASS \
        --safe_class $SAFE_CLASS \
        --guidance_scale $GUIDANCE_SCALE \
        --spatial_threshold_start $SP_START \
        --spatial_threshold_end $SP_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --use_bidirectional \
        --harmful_scale $HARMFUL_SCALE \
        --base_guidance_scale $BASE_GS \
        --restart_timestep $RT \
        --restart_guidance_fraction 0.0 \
        --restart_count 1 \
        --safety_check \
        --seed $SEED \
        --nsamples $NSAMPLES \
        --num_inference_steps $STEPS \
        --cfg_scale $CFG \
        --output_dir "${BASE_OUTPUT}/${EXP_NAME}" \
        --save_comparison
done

# ===== Experiment 6-7: With partial guidance during restart =====
for RT in 200 300; do
    EXP_NAME="restart_t${RT}_gf0.2_rc1"
    echo ""
    echo "[Exp] restart_timestep=${RT}, guidance_fraction=0.2, count=1"

    python generate_ascg_restart.py "$MODEL" \
        --prompt_file "$PROMPT_FILE" \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --gradcam_stats_file "$GRADCAM_STATS" \
        --num_classes $NUM_CLASSES \
        --harmful_class $HARMFUL_CLASS \
        --safe_class $SAFE_CLASS \
        --guidance_scale $GUIDANCE_SCALE \
        --spatial_threshold_start $SP_START \
        --spatial_threshold_end $SP_END \
        --threshold_strategy $THRESHOLD_STRATEGY \
        --use_bidirectional \
        --harmful_scale $HARMFUL_SCALE \
        --base_guidance_scale $BASE_GS \
        --restart_timestep $RT \
        --restart_guidance_fraction 0.2 \
        --restart_count 1 \
        --safety_check \
        --seed $SEED \
        --nsamples $NSAMPLES \
        --num_inference_steps $STEPS \
        --cfg_scale $CFG \
        --output_dir "${BASE_OUTPUT}/${EXP_NAME}" \
        --save_comparison
done

# ===== Experiment 8: Multiple restart cycles =====
EXP_NAME="restart_t200_gf0.0_rc2"
echo ""
echo "[Exp] restart_timestep=200, guidance_fraction=0.0, count=2"

python generate_ascg_restart.py "$MODEL" \
    --prompt_file "$PROMPT_FILE" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --gradcam_stats_file "$GRADCAM_STATS" \
    --num_classes $NUM_CLASSES \
    --harmful_class $HARMFUL_CLASS \
    --safe_class $SAFE_CLASS \
    --guidance_scale $GUIDANCE_SCALE \
    --spatial_threshold_start $SP_START \
    --spatial_threshold_end $SP_END \
    --threshold_strategy $THRESHOLD_STRATEGY \
    --use_bidirectional \
    --harmful_scale $HARMFUL_SCALE \
    --base_guidance_scale $BASE_GS \
    --restart_timestep 200 \
    --restart_guidance_fraction 0.0 \
    --restart_count 2 \
    --safety_check \
    --seed $SEED \
    --nsamples $NSAMPLES \
    --num_inference_steps $STEPS \
    --cfg_scale $CFG \
    --output_dir "${BASE_OUTPUT}/${EXP_NAME}" \
    --save_comparison

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results in: ${BASE_OUTPUT}/"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run VLM evaluation: python vlm/eval_qwen3_vl.py ..."
echo "  2. Compare SR rates between baseline and restart variants"
echo "  3. Compute FID on COCO: python eval_fid.py ..."
echo ""
