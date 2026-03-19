#!/bin/bash
# ============================================================================
# Grid Search V2: Z0 ASCG — 5-Server Multi-GPU Grid Search
#
# Designed for 5 servers × 8 GPUs each (40 GPUs total).
# Each server runs its assigned chunk of experiments.
#
# === SWEEP PARAMETERS (based on diagnostic + 3-dataset comparison) ===
#   monitoring_thr:   [0.5, 0.6]  (from coco FP/TP gap analysis)  (2)
#   sticky_trigger:   true (fixed — always on)
#   guidance_scale:   [7.5, 10, 12.5, 15, 17.5, 20]              (6)
#   grad_clip_ratio:  [0, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]   (8)
#   base_scale:       [0.0, 1.0]                                   (2)
#   spatial_threshold: 3 pairs                                      (3)
#   harm_ratio:       1.0 (fixed)
#
#   Total: 576 configs (2 × 6 × 8 × 2 × 3)
#   Per server: ~115 configs
#   Est. time: ~1h with 2 jobs/GPU (16 concurrent per server)
#
# === USAGE ===
#   # Dry run (see all configs for this server)
#   bash scripts/run_grid_v2.sh --dataset ringabell --server 0
#
#   # Run on server 0
#   bash scripts/run_grid_v2.sh --dataset ringabell --server 0 --run
#
#   # Background mode
#   bash scripts/run_grid_v2.sh --dataset ringabell --server 0 --run --nohup
#
#   # Generation only (no VLM eval)
#   bash scripts/run_grid_v2.sh --dataset ringabell --server 0 --run --gen-only --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
JOBS_PER_GPU=2
DATASET=""
SERVER_ID=-1
NUM_SERVERS=5
GEN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --server) SERVER_ID="$2"; shift 2;;
        --num-servers) NUM_SERVERS="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        --gen-only) GEN_ONLY=true; shift;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset required (ringabell / unlearndiff / mma)"
    exit 1
fi
if [ "$SERVER_ID" -lt 0 ]; then
    echo "ERROR: --server required (0..4)"
    exit 1
fi

# ============================================
# Dataset Config
# ============================================
case $DATASET in
    ringabell)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv";;
    unlearndiff)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv";;
    mma)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv";;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"; exit 1;;
esac

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1
fi

# Nohup
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./grid_v2_output/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/grid_v2_${DATASET}_server${SERVER_ID}_${TIMESTAMP}.log"
    echo "Running server ${SERVER_ID} in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    EXTRA=""
    if [ "$GEN_ONLY" = true ]; then EXTRA="$EXTRA --gen-only"; fi
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --server "$SERVER_ID" --num-servers "$NUM_SERVERS" --run --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# Fixed Config
# ============================================
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS_PATH="./harmful_stats.pt"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_monitoring.py"

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"
GRADCAM_LAYER="layer2"
HARM_RATIO=1.0

# ============================================
# GRID SEARCH PARAMETERS
# ============================================

# Monitoring: 3-dataset comparison에서 FP 낮고 gap 최대인 threshold
# softmax_p thr=0.50: FP=12%, avg_GAP=+62.4%
# softmax_p thr=0.60: FP=8%, avg_GAP=+63.6%
MONITORING_THRESHOLDS=(0.5 0.6)
# Sticky 항상 켜기 (trigger 이후 모든 step guidance)
STICKY="true"
# gs=7.5~20 (2.5 단위), 그 이상은 clipping 때문에 의미 없음
GUIDANCE_SCALES=(7.5 10.0 12.5 15.0 17.5 20.0)
# cl=0.1~1.0 + 0(no clip). 2.0 이상은 artifact 심함
GRAD_CLIP_RATIOS=(0 0.1 0.15 0.2 0.3 0.5 0.75 1.0)
BASE_SCALES=(0.0 1.0)
SPATIAL_THRESHOLDS=(
    "0.1 0.3"
    "0.2 0.3"
    "0.3 0.3"
)

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_v2_output/${DATASET}"

# ============================================
# Build ALL combinations
# ============================================
declare -a ALL_COMBINATIONS=()
for mt in "${MONITORING_THRESHOLDS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for cl in "${GRAD_CLIP_RATIOS[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SPATIAL_THRESHOLDS[@]}"; do
                    ALL_COMBINATIONS+=("${mt}|${gs}|${cl}|${bs}|${sp}")
                done
            done
        done
    done
done

TOTAL_ALL=${#ALL_COMBINATIONS[@]}

# ============================================
# Split for this server
# ============================================
CHUNK_SIZE=$(( (TOTAL_ALL + NUM_SERVERS - 1) / NUM_SERVERS ))
START=$(( SERVER_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if [ $END -gt $TOTAL_ALL ]; then END=$TOTAL_ALL; fi

declare -a COMBINATIONS=()
for (( i=START; i<END; i++ )); do
    COMBINATIONS+=("${ALL_COMBINATIONS[$i]}")
done

TOTAL=${#COMBINATIONS[@]}

echo "=============================================="
echo "Z0 ASCG GRID SEARCH V2 — SERVER ${SERVER_ID}/${NUM_SERVERS}"
echo "=============================================="
echo "Dataset: $DATASET"
echo "This server: configs [${START}:${END}] = ${TOTAL} / ${TOTAL_ALL} total"
echo "mon_thr: ${MONITORING_THRESHOLDS[*]}"
echo "sticky: always ON"
echo "gs: ${GUIDANCE_SCALES[*]}"
echo "grad_clip: ${GRAD_CLIP_RATIOS[*]}"
echo "bs: ${BASE_SCALES[*]}"
echo "spatial: ${#SPATIAL_THRESHOLDS[@]} pairs"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
echo "Gen only: $GEN_ONLY"
echo "=============================================="

# ============================================
# Dry run
# ============================================
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Server ${SERVER_ID} experiments:"
    for (( i=0; i<TOTAL; i++ )); do
        combo="${COMBINATIONS[$i]}"
        IFS='|' read -r MT GS CL BS SP <<< "$combo"
        SP_START=$(echo $SP | awk '{print $1}')
        SP_END=$(echo $SP | awk '{print $2}')
        GPU_IDX=$((i % NUM_GPUS))

        EXP_NAME="mon${MT}_gs${GS}_cl${CL}_sticky_bs${BS}_sp${SP_START}-${SP_END}"
        echo "  [${i}] GPU${GPU_IDX} | ${EXP_NAME}"
    done
    echo ""
    echo "Server ${SERVER_ID}: ${TOTAL} experiments on ${NUM_GPUS} GPUs"
    echo "All servers combined: ${TOTAL_ALL} experiments"
    echo "Add --run to execute."
    exit 0
fi

# ============================================
# GPU setup
# ============================================
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
echo "GPUs: ${GPU_LIST[*]}"
mkdir -p "${OUTPUT_BASE}/logs"

# ============================================
# Run function
# ============================================
run_experiment() {
    local GPU_IDX=$1
    local MON_THR=$2
    local GS=$3
    local CLIP=$4
    local BS=$5
    local SP_START=$6
    local SP_END=$7

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}

    # Build experiment name
    local EXP_NAME="mon${MON_THR}_gs${GS}_cl${CLIP}_sticky_bs${BS}_sp${SP_START}-${SP_END}"

    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
        return 0
    fi

    echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Step 1: Generate
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        local GEN_CMD="CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${GEN_SCRIPT} \
            --ckpt_path ${SD_CKPT} \
            --prompt_file ${PROMPT_FILE} \
            --output_dir ${OUTPUT_DIR} \
            --classifier_ckpt ${CLASSIFIER_CKPT} \
            --harmful_stats_path ${HARMFUL_STATS_PATH} \
            --gradcam_layer ${GRADCAM_LAYER} \
            --monitoring_mode classifier \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --harm_ratio ${HARM_RATIO} \
            --grad_clip_ratio ${CLIP} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy ${THRESHOLD_STRATEGY} \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES}"

        GEN_CMD="${GEN_CMD} --monitoring_threshold ${MON_THR} --sticky_trigger"

        eval ${GEN_CMD} >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] FAILED gen: ${EXP_NAME}"
            return 1
        fi
    else
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] SKIP gen (done): ${EXP_NAME}"
    fi

    # Step 2: VLM eval
    if [ "$GEN_ONLY" = true ]; then
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] GEN DONE: ${EXP_NAME}"
        return 0
    fi

    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

    if [ $? -eq 0 ]; then
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] DONE: ${EXP_NAME}"
    else
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] FAILED eval: ${EXP_NAME}"
        return 1
    fi
    return 0
}

# ============================================
# Main loop
# ============================================
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS
IDX=0

for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r MT GS CL BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MT" "$GS" "$CL" "$BS" "$SP_START" "$SP_END" &
                RUNNING_JOBS[$slot]=$!
                IDX=$((IDX + 1))
                echo "[Server ${SERVER_ID}] Progress: ${IDX}/${TOTAL}"
                sleep 1
                break 2
            fi
        done
        sleep 5
    done
done

echo ""
echo "Waiting for remaining jobs..."
wait

echo ""
echo "=============================================="
echo "SERVER ${SERVER_ID} COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Experiments: $TOTAL"
echo "Results: ${OUTPUT_BASE}/"

# Quick summary
COMPLETED=0
GEN_DONE=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r MT GS CL BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')

    EXP_NAME="mon${MT}_gs${GS}_cl${CL}_sticky_bs${BS}_sp${SP_START}-${SP_END}"

    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    elif [ -f "${OUTPUT_BASE}/${EXP_NAME}/generation_stats.json" ]; then
        GEN_DONE=$((GEN_DONE + 1))
    fi
done
echo "Fully completed: ${COMPLETED}/${TOTAL}"
if [ $GEN_DONE -gt 0 ]; then
    echo "Gen done (eval pending): ${GEN_DONE}"
fi
echo "Done!"
