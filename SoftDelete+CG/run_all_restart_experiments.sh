#!/bin/bash
# =============================================================================
# Master script: Grid Search → COCO Eval → Analysis (all automated)
#
# This script runs everything sequentially:
#   1. Extended grid search on Ring-A-Bell (34 configs, 4 GPUs)
#   2. COCO FP/FID evaluation on best configs (4 GPUs)
#   3. Result analysis and summary
#
# Already running as background - will complete unattended.
# =============================================================================

set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="scg_outputs/restart_grid/master_log_${TIMESTAMP}.txt"
mkdir -p scg_outputs/restart_grid

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=============================================="
echo "MASTER EXPERIMENT PIPELINE"
echo "Started: $(date)"
echo "=============================================="

# =============================================================================
# STEP 1: Grid search is already running (brml09q8d)
# Skip to STEP 2 when grid search results exist
# =============================================================================

echo ""
echo "[Step 1] Waiting for grid search to complete..."

# Wait until all 34 experiments have 79 images
TARGET_DIR="scg_outputs/restart_grid/ringabell"
TOTAL_EXPS=34

while true; do
    completed=0
    for d in "$TARGET_DIR"/*/; do
        [ -d "$d" ] || continue
        count=$(ls "$d"/*.png 2>/dev/null | wc -l)
        if [ "$count" -ge 79 ]; then
            completed=$((completed + 1))
        fi
    done

    if [ "$completed" -ge "$TOTAL_EXPS" ]; then
        echo "[Step 1] Grid search complete! ($completed/$TOTAL_EXPS)"
        break
    fi

    echo "  $(date +%H:%M:%S) - $completed/$TOTAL_EXPS experiments done, waiting..."
    sleep 60
done

# =============================================================================
# STEP 2: Run initial analysis on grid search results
# =============================================================================

echo ""
echo "[Step 2] Running analysis on grid search results..."

python analyze_restart_results.py \
    --results_dir "$TARGET_DIR" \
    --export_csv "scg_outputs/restart_grid/results_summary.csv"

# =============================================================================
# STEP 3: COCO evaluation for FID/FP measurement
# =============================================================================

echo ""
echo "[Step 3] Running COCO evaluation for FID/FP..."

MODEL="CompVis/stable-diffusion-v1-4"
COCO_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"
CLF="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json"
COCO_BASE="scg_outputs/restart_grid/coco"
COCO_LOG="$COCO_BASE/logs"
mkdir -p "$COCO_LOG"

NUM_GPUS=4
END_IDX=50

# Key configs for COCO eval
declare -a COCO_CONFIGS
COCO_CONFIGS+=("baseline|0|0.0|0")
COCO_CONFIGS+=("restart_t100|100|0.0|1")
COCO_CONFIGS+=("restart_t150|150|0.0|1")
COCO_CONFIGS+=("restart_t200|200|0.0|1")
COCO_CONFIGS+=("restart_t250|250|0.0|1")
COCO_CONFIGS+=("restart_t300|300|0.0|1")
COCO_CONFIGS+=("restart_t400|400|0.0|1")
COCO_CONFIGS+=("restart_t500|500|0.0|1")
COCO_CONFIGS+=("restart_t200_gf0.1|200|0.1|1")
COCO_CONFIGS+=("restart_t300_gf0.1|300|0.1|1")
COCO_CONFIGS+=("restart_t200_rc2|200|0.0|2")
COCO_CONFIGS+=("restart_t300_rc2|300|0.0|2")

TOTAL_COCO=${#COCO_CONFIGS[@]}
echo "  COCO configs: $TOTAL_COCO"

IDX=0
while [ $IDX -lt $TOTAL_COCO ]; do
    PIDS=()
    GPU=0

    BATCH_END=$((IDX + NUM_GPUS))
    if [ $BATCH_END -gt $TOTAL_COCO ]; then BATCH_END=$TOTAL_COCO; fi

    echo "  COCO batch: experiments $((IDX+1))-${BATCH_END}/${TOTAL_COCO}"

    while [ $IDX -lt $BATCH_END ]; do
        IFS='|' read -r NAME RT GF RC <<< "${COCO_CONFIGS[$IDX]}"

        echo "    [GPU $GPU] $NAME"

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
            --seed 42 --nsamples 1 --num_inference_steps 50 --cfg_scale 7.5 \
            --start_idx 0 --end_idx $END_IDX \
            --output_dir "$COCO_BASE/$NAME" \
            > "$COCO_LOG/${NAME}.log" 2>&1 &

        PIDS+=($!)
        GPU=$((GPU + 1))
        IDX=$((IDX + 1))
    done

    for PID in "${PIDS[@]}"; do wait $PID; done
    echo "  COCO batch complete!"
done

echo "[Step 3] COCO evaluation complete!"

# =============================================================================
# STEP 4: Final summary
# =============================================================================

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  Ring-A-Bell: scg_outputs/restart_grid/ringabell/"
echo "  COCO:        scg_outputs/restart_grid/coco/"
echo "  Summary CSV: scg_outputs/restart_grid/results_summary.csv"
echo "  Master log:  $MASTER_LOG"
echo ""
echo "Ring-A-Bell experiment summary:"

# Quick count
for d in "$TARGET_DIR"/*/; do
    name=$(basename "$d")
    imgs=$(ls "$d"/*.png 2>/dev/null | wc -l)
    fb=$(python3 -c "import json; d=json.load(open('$d/generation_stats.json')); print(d['summary']['safety_fallback'])" 2>/dev/null || echo "?")
    echo "  $name: ${imgs} imgs, ${fb} fallbacks"
done

echo ""
echo "COCO experiment summary:"
for d in "$COCO_BASE"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [ "$name" = "logs" ] && continue
    imgs=$(ls "$d"/*.png 2>/dev/null | wc -l)
    echo "  $name: ${imgs} imgs"
done
