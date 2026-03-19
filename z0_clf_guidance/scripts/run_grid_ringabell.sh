#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Grid Search: Z0 CLF Guidance + Gaussian CDF Spatial CG on Ring-A-Bell
# ═══════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_grid_ringabell.sh                      # 1 GPU, dry run
#   bash scripts/run_grid_ringabell.sh --run                # 1 GPU, real run
#   bash scripts/run_grid_ringabell.sh --run --num_gpus 8 --gpu_ids 0,1,2,3,4,5,6,7
#
# Default sweep (28 experiments, deduplicated):
#   guidance_scales:       [5.0, 10.0, 15.0, 20.0]
#   spatial_modes:         [none, gradcam]
#   spatial_thresholds:    [0.3, 0.5, 0.7]  (CDF percentile)
#   spatial_soft_options:  [0, 1]
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Navigate to project root ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ──
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS_PATH="./harmful_stats.pt"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
CKPT_PATH="CompVis/stable-diffusion-v1-4"
NUM_GPUS=1
GPU_IDS=""
DRY_RUN="--dry_run"
EXTRA_ARGS=""

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            DRY_RUN=""
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --classifier_ckpt)
            CLASSIFIER_CKPT="$2"
            shift 2
            ;;
        --harmful_stats_path)
            HARMFUL_STATS_PATH="$2"
            shift 2
            ;;
        --prompt_file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ── Validate ──
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "ERROR: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    echo "Available checkpoints:"
    ls ./work_dirs/z0_resnet18_classifier/checkpoint/ 2>/dev/null | tail -5 || echo "  (none)"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$HARMFUL_STATS_PATH" ]; then
    echo "WARNING: Harmful stats not found: $HARMFUL_STATS_PATH"
    echo "  Spatial thresholding will use max-normalization fallback."
    echo "  Run compute_harmful_stats.py first for CDF-based thresholding."
    HARMFUL_STATS_ARG=""
else
    HARMFUL_STATS_ARG="--harmful_stats_path $HARMFUL_STATS_PATH"
fi

# ── Build GPU argument ──
GPU_ARG=""
if [ -n "$GPU_IDS" ]; then
    GPU_ARG="--gpu_ids $GPU_IDS"
fi

# ── Output directory ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="./grid_search_output/ringabell_${TIMESTAMP}"

# ── Summary ──
echo "═══════════════════════════════════════════════════════════════"
echo "  Z0 CLF Guidance - Gaussian CDF Spatial Grid Search (Ring-A-Bell)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Classifier:     $CLASSIFIER_CKPT"
echo "  Harmful stats:  $HARMFUL_STATS_PATH"
echo "  Prompt file:    $PROMPT_FILE"
echo "  SD model:       $CKPT_PATH"
echo "  GPUs:           $NUM_GPUS"
echo "  Output:         $OUTPUT_ROOT"
if [ -n "$DRY_RUN" ]; then
    echo "  Mode:           DRY RUN (add --run to execute)"
fi
echo "═══════════════════════════════════════════════════════════════"

# ── Run ──
CMD="python grid_search_spatial_cg.py \
    --ckpt_path $CKPT_PATH \
    --classifier_ckpt $CLASSIFIER_CKPT \
    $HARMFUL_STATS_ARG \
    --prompt_file $PROMPT_FILE \
    --output_root $OUTPUT_ROOT \
    --num_gpus $NUM_GPUS \
    $GPU_ARG \
    --guidance_mode safe_minus_harm \
    --safe_classes 1 \
    --harm_classes 2 \
    --guidance_scales 5.0 10.0 15.0 20.0 \
    --spatial_modes none gradcam \
    --spatial_thresholds 0.3 0.5 0.7 \
    --spatial_soft_options 0 1 \
    $DRY_RUN \
    $EXTRA_ARGS"

if [ -n "$DRY_RUN" ]; then
    # Dry run: just print
    eval $CMD
else
    # Real run: use nohup for background execution
    LOG_FILE="${OUTPUT_ROOT}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo ""
    echo "Launching in background..."
    echo "  Log: $LOG_FILE"
    echo "  Monitor: tail -f $LOG_FILE"
    echo ""
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo ""
    echo "To check progress:"
    echo "  tail -f $LOG_FILE"
    echo "  ls $OUTPUT_ROOT/ | wc -l    # count completed experiments"
fi
