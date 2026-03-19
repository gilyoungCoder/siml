#!/bin/bash
# =============================================================================
# ASCG + Restart - COCO evaluation for FID/FP measurement
#
# Generates 50 COCO images per config to measure:
#   - FP rate (false positive guidance on benign prompts)
#   - FID (image quality compared to baseline)
#
# Runs the best restart configs identified from ringabell experiments.
# =============================================================================

set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

MODEL="CompVis/stable-diffusion-v1-4"
COCO_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"
CLF="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json"
BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/restart_grid/coco"
LOGDIR="${BASE}/logs"
mkdir -p "$LOGDIR"

# Use first 50 COCO prompts
END_IDX=50

# Configs to test (baseline + best restart variants)
declare -a CONFIGS
CONFIGS+=("baseline|0|0.0|0")
CONFIGS+=("restart_t100|100|0.0|1")
CONFIGS+=("restart_t200|200|0.0|1")
CONFIGS+=("restart_t300|300|0.0|1")
CONFIGS+=("restart_t400|400|0.0|1")
CONFIGS+=("restart_t200_gf0.1|200|0.1|1")
CONFIGS+=("restart_t200_rc2|200|0.0|2")
CONFIGS+=("restart_t300_gf0.1|300|0.1|1")

echo "=============================================="
echo "COCO FID/FP Evaluation (${#CONFIGS[@]} configs)"
echo "=============================================="

NUM_GPUS=4
PIDS=()
GPU=0
BATCH=0
for CFG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME RT GF RC <<< "$CFG"

    echo "[GPU $GPU] $NAME"

    CUDA_VISIBLE_DEVICES=$GPU python generate_ascg_restart.py "$MODEL" \
        --prompt_file "$COCO_FILE" \
        --classifier_ckpt "$CLF" \
        --gradcam_stats_file "$STATS" \
        --num_classes 4 --harmful_class 2 --safe_class 1 \
        --guidance_scale 10.0 \
        --spatial_threshold_start 0.3 --spatial_threshold_end 0.5 \
        --threshold_strategy cosine_anneal \
        --use_bidirectional --harmful_scale 1.0 --base_guidance_scale 0.0 \
        --restart_timestep $RT \
        --restart_guidance_fraction $GF \
        --restart_count $RC \
        --safety_check \
        --seed 42 --nsamples 1 --num_inference_steps 50 --cfg_scale 7.5 \
        --start_idx 0 --end_idx $END_IDX \
        --output_dir "${BASE}/${NAME}" \
        > "${LOGDIR}/${NAME}.log" 2>&1 &

    PIDS+=($!)
    GPU=$((GPU + 1))

    # If all GPUs used, wait for batch
    if [ $GPU -ge $NUM_GPUS ]; then
        echo "  Waiting for batch..."
        for PID in "${PIDS[@]}"; do wait $PID; done
        PIDS=()
        GPU=0
    fi
done

# Wait for final batch
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for final batch..."
    for PID in "${PIDS[@]}"; do wait $PID; done
fi

echo ""
echo "COCO generation complete! Output: ${BASE}/"
echo "Next: compute FID with eval_fid.py"
