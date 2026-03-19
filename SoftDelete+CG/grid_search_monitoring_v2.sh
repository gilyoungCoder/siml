#!/bin/bash
# ============================================================================
# Grid Search v2: Multiple "When" Monitoring Ideas
#
# Tests 5 different monitoring approaches for ASCG:
#   1. anchored_ssscore: SSScore with anchor="a person wearing clothes"
#   2. noise_div: Online noise divergence (extra UNet fwd with safe prompt)
#   3. noise_div_free: CFG magnitude proxy (no extra UNet cost)
#   4. ssscore_weighted: SSScore as continuous guidance multiplier
#   5. grad_norm: Classifier gradient norm as monitoring signal
#
# Each has a sticky variant. Focused on proven good spatial settings.
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
SITE=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required (ringabell / coco)"
    exit 1
fi

# Dataset config
case $DATASET in
    ringabell)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
        END_IDX=-1
        ;;
    coco)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"
        END_IDX=50
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET' (use ringabell or coco)"; exit 1;;
esac

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    SITE_TAG=""
    if [ -n "$SITE" ]; then SITE_TAG="_site${SITE}"; fi
    LOG_FILE="${LOG_DIR}/monitoring_v2_grid_${DATASET}${SITE_TAG}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    SITE_ARG=""
    if [ -n "$SITE" ]; then SITE_ARG="--site $SITE"; fi
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --run --num-gpus "$NUM_GPUS" $SITE_ARG > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_unified_monitoring.py"

# Fixed params (proven good from previous grids)
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

HARMFUL_KW="nude naked breast topless bare undress nsfw"

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/monitoring_v2_grid/${DATASET}"

# ============================================
# BUILD EXPERIMENT COMBINATIONS
# Format: MON_MODE|THRESHOLD|GS|BS|SP_S|SP_E|EXTRA_ARGS
# ============================================
declare -a ALL_COMBOS=()

# --- Best known spatial settings (from previous grid: gs=10, bs=3 is best) ---
GS=10.0
BS=3.0

# Two spatial configs to test
SP_PAIRS=("0.2 0.4" "0.3 0.5")

# ============================================
# 1. Anchored SSScore (anchor="a person wearing clothes")
# ============================================
SSSCORE_CACHE_ANCHORED="${OUTPUT_BASE}/ssscore_cache_anchored.json"

for thr in 0.3 0.4 0.5 0.6 0.7; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("ssscore_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|ssscore_anchored")
    done
done
# 5 thresholds × 2 spatial = 10 experiments

# ============================================
# 2. Noise Divergence (online, image-specific)
# ============================================
# Divergence values are L2 norms; need to calibrate thresholds
# Typical noise tensor is 4×64×64, so L2 norms ~ 50-200
for thr in 30.0 50.0 70.0 100.0 150.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|noise_div_safe")
    done
done
# 5 × 2 = 10

# ============================================
# 3. Noise Divergence Free (CFG magnitude proxy, no extra cost)
# ============================================
for thr in 30.0 50.0 70.0 100.0 150.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_free_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 5 × 2 = 10

# ============================================
# 4. SSScore Weighted (continuous multiplier, anchored)
# ============================================
SSSCORE_CACHE_WEIGHTED="${OUTPUT_BASE}/ssscore_cache_weighted.json"

for wmin in 0.0 0.3 0.5; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("ssscore_weighted|wmin${wmin}|${GS}|${BS}|${SP_S}|${SP_E}|ssscore_weighted_${wmin}")
    done
done
# 3 × 2 = 6

# ============================================
# 5. Gradient Norm (classifier sensitivity)
# ============================================
for thr in 0.5 1.0 2.0 5.0 10.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("grad_norm_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 5 × 2 = 10

# ============================================
# 6. Non-sticky variants for best thresholds
# ============================================
for thr in 50.0 70.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|noise_div_safe")
    done
done
# 2 × 2 = 4

for thr in 50.0 70.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_free|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 2 × 2 = 4

for thr in 1.0 2.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("grad_norm|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 2 × 2 = 4

# ============================================
# 7. Extra: higher guidance scale for noise_div
# ============================================
for thr in 50.0 70.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_sticky|${thr}|20.0|${BS}|${SP_S}|${SP_E}|noise_div_safe")
    done
done
# 2 × 2 = 4

# Total: 10 + 10 + 10 + 6 + 10 + 4 + 4 + 4 + 4 = 62 experiments

TOTAL=${#ALL_COMBOS[@]}

# Site splitting
SPLIT=$(( TOTAL * 3 / 5 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$SPLIT
elif [ "$SITE" = "B" ]; then
    START=$SPLIT; END=$TOTAL
else
    START=0; END=$TOTAL
fi
SITE_COUNT=$((END - START))

echo "=============================================="
echo "MONITORING V2 GRID SEARCH"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Total configs: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Methods: anchored_ssscore, noise_div, noise_div_free, ssscore_weighted, grad_norm"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E EXTRA <<< "$combo"
        echo "  [$ci] ${MMODE}_thr${THR}_gs${GS_VAL}_bs${BS_VAL}_sp${SP_S}-${SP_E}"
    done
    echo ""
    echo "Total: $SITE_COUNT experiments on $NUM_GPUS GPUs"
    echo "Add --run to execute."
    exit 0
fi

# ============================================
# GPU SETUP
# ============================================

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi

declare -A GPU_PIDS
for g in "${GPU_LIST[@]}"; do GPU_PIDS[$g]=0; done

wait_for_gpu() {
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                GPU_PIDS[$g]=0
                echo "$g"
                return
            fi
        done
        sleep 5
    done
}

run_experiment() {
    local GPU_ID=$1
    local MMODE=$2
    local THR=$3
    local GS_VAL=$4
    local BS_VAL=$5
    local SP_S=$6
    local SP_E=$7
    local EXTRA=$8

    local EXP_NAME="${MMODE}_thr${THR}_gs${GS_VAL}_bs${BS_VAL}_sp${SP_S}-${SP_E}"
    local OUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_DIR="${OUTPUT_BASE}/logs"
    mkdir -p "$LOG_DIR"

    # Skip if already completed
    if [ -f "${OUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
        echo "[GPU $GPU_ID] SKIP (done): $EXP_NAME"
        return
    fi

    # Count existing images
    local IMG_COUNT=0
    if [ -d "${OUT_DIR}/images" ]; then
        IMG_COUNT=$(ls "${OUT_DIR}/images/"*.png 2>/dev/null | wc -l)
    fi

    echo "[GPU $GPU_ID] START: $EXP_NAME"

    (
        SAFE_PROMPT="a person wearing clothes"

        # Build base args array
        declare -a CMD_ARGS=(
            python3 "$GEN_SCRIPT"
            --ckpt_path "$SD_CKPT"
            --prompt_file "$PROMPT_FILE"
            --output_dir "${OUT_DIR}"
            --classifier_ckpt "$CLASSIFIER_CKPT"
            --gradcam_stats_dir "$GRADCAM_STATS_DIR"
            --monitoring_mode "$MMODE"
            --monitoring_threshold "$THR"
            --guidance_scale "$GS_VAL"
            --base_guidance_scale "$BS_VAL"
            --spatial_threshold_start "$SP_S"
            --spatial_threshold_end "$SP_E"
            --spatial_threshold_strategy "$THRESHOLD_STRATEGY"
            --spatial_mode gradcam
            --num_inference_steps "$NUM_STEPS"
            --nsamples "$NSAMPLES"
            --seed "$SEED"
            --cfg_scale "$CFG_SCALE"
            --start_idx 0
            --end_idx "$END_IDX"
            --harmful_keywords $HARMFUL_KW
        )

        # Add extra args based on tag
        case "$EXTRA" in
            ssscore_anchored)
                CMD_ARGS+=(--ssscore_anchor "$SAFE_PROMPT" --ssscore_cache "${SSSCORE_CACHE_ANCHORED}")
                ;;
            noise_div_safe)
                CMD_ARGS+=(--noise_div_safe_prompt "$SAFE_PROMPT")
                ;;
            ssscore_weighted_*)
                WMIN=$(echo "$EXTRA" | sed 's/ssscore_weighted_//')
                CMD_ARGS+=(--ssscore_anchor "$SAFE_PROMPT" --ssscore_cache "${SSSCORE_CACHE_WEIGHTED}" --ssscore_weight_min "${WMIN}" --monitoring_threshold 0.0)
                ;;
        esac

        # Step 1: Generate images (skip if already done)
        if [ "$IMG_COUNT" -lt 79 ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "${CMD_ARGS[@]}" 2>&1
        fi

        # Step 2: VLM eval
        if [ ! -f "${OUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
            eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
            CUDA_VISIBLE_DEVICES=$GPU_ID python3 $VLM_SCRIPT \
                "${OUT_DIR}" nudity qwen \
                2>&1
            eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
        fi

    ) > "${LOG_DIR}/${EXP_NAME}.log" 2>&1 &

    GPU_PIDS[$GPU_ID]=$!
}

# ============================================
# MAIN LOOP
# ============================================

echo ""
echo "Starting experiments..."
echo ""

COMPLETED=0
for (( ci=START; ci<END; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E EXTRA <<< "$combo"

    GPU=$(wait_for_gpu)
    COMPLETED=$((COMPLETED + 1))
    echo "Progress: $COMPLETED/$SITE_COUNT [combo $ci]"

    run_experiment "$GPU" "$MMODE" "$THR" "$GS_VAL" "$BS_VAL" "$SP_S" "$SP_E" "$EXTRA"
done

echo ""
echo "Waiting for remaining jobs..."
wait

echo ""
echo "=============================================="
echo "MONITORING V2 GRID SEARCH COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Results: $OUTPUT_BASE"
echo "Completed: $SITE_COUNT/$TOTAL"
echo ""

# Summary
echo "=== TOP RESULTS ==="
for d in ${OUTPUT_BASE}/*/; do
    f="${d}results_qwen3_vl_nudity.txt"
    if [ -f "$f" ]; then
        sr=$(grep "SR " "$f" | grep -oP '[\d.]+%' | head -1)
        name=$(basename "$d")
        echo "SR=${sr} ${name}"
    fi
done | sort -t= -k2 -rn | head -20

echo ""
echo "Done!"
