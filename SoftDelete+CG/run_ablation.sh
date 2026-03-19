#!/bin/bash
# ============================================================================
# Ablation Study: 4 conditions × multiple guidance scales × 4 datasets
# Conditions:
#   1. Full CG (no monitoring, no spatial mask)
#   2. Full CG + Monitoring (monitoring, no spatial mask)
#   3. Spatial CG (no monitoring, spatial mask)
#   4. Ours (monitoring + spatial mask) — already done (mon4class)
#
# For each condition, try multiple guidance scales to find best SR
# without destroying image quality.
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

CLASSIFIER_CKPT="work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"
ALWAYS_SCRIPT="generate_nudity_4class_spatial_cg_always.py"
MON_SCRIPT="generate_nudity_4class_monitoring_early_exit.py"

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5

# Ours best config for monitoring threshold / spatial
MON_THRESH=0.05
SP_START=0.2
SP_END=0.3
SP_STRATEGY="cosine"
TEXT_EXIT_THRESH=0.50

OUTPUT_BASE="scg_outputs/ablation_early_exit"

declare -A DATASET_PROMPTS
DATASET_PROMPTS[p4dn]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
DATASET_PROMPTS[ringabell]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
DATASET_PROMPTS[unlearndiff]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
DATASET_PROMPTS[mma]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"

# Per-condition guidance scales
declare -A COND_GS
COND_GS[full_cg]="2"
COND_GS[full_cg_mon]="2 3"
COND_GS[spatial_cg]="5"

# Parse args
USE_NOHUP=false
NUM_GPUS=8
DATASETS="p4dn ringabell unlearndiff mma"
CONDITIONS="full_cg full_cg_mon spatial_cg"

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        --conditions) CONDITIONS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

SCRIPT_PATH="/mnt/home/yhgil99/unlearning/SoftDelete+CG/run_ablation.sh"
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${OUTPUT_BASE}/ablation_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Running in background... Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" --num-gpus "$NUM_GPUS" --datasets "$DATASETS" --conditions "$CONDITIONS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
echo "GPUs: ${GPU_LIST[*]}"

# ============================================
# Collect all jobs
# ============================================
declare -a JOBS=()  # "condition|dataset|gs"

for ds in $DATASETS; do
    for cond in $CONDITIONS; do
        for gs in ${COND_GS[$cond]}; do
            DIR="${OUTPUT_BASE}/${cond}/${ds}/gs${gs}"
            if [ ! -f "${DIR}/generation_stats.json" ]; then
                JOBS+=("${cond}|${ds}|${gs}")
            fi
        done
    done
done

TOTAL=${#JOBS[@]}
echo "=============================================="
echo "ABLATION STUDY"
echo "Total jobs: $TOTAL"
echo "=============================================="

if [ $TOTAL -eq 0 ]; then
    echo "Nothing to generate!"
    exit 0
fi

# ============================================
# Run function
# ============================================
run_job() {
    local GPU_IDX=$1 COND=$2 DS=$3 GS=$4
    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local PROMPT_FILE="${DATASET_PROMPTS[$DS]}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${COND}/${DS}/gs${GS}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${COND}_${DS}_gs${GS}.log"

    if [ -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP: ${COND}/${DS}/gs${GS}"
        return 0
    fi

    mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"
    echo "[GPU ${ACTUAL_GPU}] RUN: ${COND}/${DS}/gs${GS}"

    case "$COND" in
        full_cg)
            # No monitoring, no spatial mask: threshold=0 (always apply), base=gs, spatial=0
            CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${MON_SCRIPT}" \
                --ckpt_path "${SD_CKPT}" \
                --prompt_file "${PROMPT_FILE}" \
                --output_dir "${OUTPUT_DIR}" \
                --classifier_ckpt "${CLASSIFIER_CKPT}" \
                --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
                --monitoring_threshold 0 \
                --guidance_scale ${GS} \
                --base_guidance_scale ${GS} \
                --spatial_threshold_start 0.0 \
                --spatial_threshold_end 0.0 \
                --spatial_threshold_strategy constant \
                --num_inference_steps ${NUM_STEPS} \
                --cfg_scale ${CFG_SCALE} \
                --seed ${SEED} \
                --nsamples ${NSAMPLES} \
                --text_exit_threshold ${TEXT_EXIT_THRESH} \
                >> "${LOG_FILE}" 2>&1
            ;;
        full_cg_mon)
            # Monitoring, no spatial mask: base=gs, spatial=0
            CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${MON_SCRIPT}" \
                --ckpt_path "${SD_CKPT}" \
                --prompt_file "${PROMPT_FILE}" \
                --output_dir "${OUTPUT_DIR}" \
                --classifier_ckpt "${CLASSIFIER_CKPT}" \
                --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
                --monitoring_threshold ${MON_THRESH} \
                --guidance_scale ${GS} \
                --base_guidance_scale ${GS} \
                --spatial_threshold_start 0.0 \
                --spatial_threshold_end 0.0 \
                --spatial_threshold_strategy constant \
                --num_inference_steps ${NUM_STEPS} \
                --cfg_scale ${CFG_SCALE} \
                --seed ${SEED} \
                --nsamples ${NSAMPLES} \
                --text_exit_threshold ${TEXT_EXIT_THRESH} \
                >> "${LOG_FILE}" 2>&1
            ;;
        spatial_cg)
            # No monitoring, spatial mask: threshold=0 (always apply), spatial params
            CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${MON_SCRIPT}" \
                --ckpt_path "${SD_CKPT}" \
                --prompt_file "${PROMPT_FILE}" \
                --output_dir "${OUTPUT_DIR}" \
                --classifier_ckpt "${CLASSIFIER_CKPT}" \
                --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
                --monitoring_threshold 0 \
                --guidance_scale ${GS} \
                --base_guidance_scale 2.0 \
                --spatial_threshold_start ${SP_START} \
                --spatial_threshold_end ${SP_END} \
                --spatial_threshold_strategy cosine_anneal \
                --num_inference_steps ${NUM_STEPS} \
                --cfg_scale ${CFG_SCALE} \
                --seed ${SEED} \
                --nsamples ${NSAMPLES} \
                --text_exit_threshold ${TEXT_EXIT_THRESH} \
                >> "${LOG_FILE}" 2>&1
            ;;
    esac

    if [ $? -ne 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${COND}/${DS}/gs${GS}"
    else
        echo "[GPU ${ACTUAL_GPU}] DONE: ${COND}/${DS}/gs${GS}"
    fi
}

# ============================================
# Dispatch
# ============================================
PIDS=()
GPU_IDX=0

for job in "${JOBS[@]}"; do
    IFS='|' read -r COND DS GS <<< "$job"
    run_job $((GPU_IDX % ${#GPU_LIST[@]})) "$COND" "$DS" "$GS" &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    if [ $((GPU_IDX % ${#GPU_LIST[@]})) -eq 0 ] && [ ${#PIDS[@]} -ge ${#GPU_LIST[@]} ]; then
        echo "--- Waiting for batch (${#PIDS[@]} jobs) ---"
        for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
        PIDS=()
    fi
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for final batch ---"
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
fi

echo ""
echo "=== Ablation generation complete ==="
echo "Condition 4 (Ours) already in: fine_grid_mon4class/{ds}/mon0.05_gs12.5_bs2.0_sp0.2-0.3"
