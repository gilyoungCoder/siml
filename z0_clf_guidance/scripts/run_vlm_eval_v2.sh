#!/bin/bash
# ============================================================================
# VLM Eval V2: Evaluate grid_v2 generated images with Qwen3-VL
#
# Splits all experiment dirs across 5 servers × 8 GPUs.
#
# === USAGE ===
#   bash scripts/run_vlm_eval_v2.sh --dataset ringabell --server 0 --run --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"

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

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --server) SERVER_ID="$2"; shift 2;;
        --num-servers) NUM_SERVERS="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset required"; exit 1
fi
if [ "$SERVER_ID" -lt 0 ]; then
    echo "ERROR: --server required (0..4)"; exit 1
fi

OUTPUT_BASE="./grid_v2_output/${DATASET}"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

# Nohup
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${OUTPUT_BASE}/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/vlm_eval_server${SERVER_ID}_${TIMESTAMP}.log"
    echo "Running VLM eval server ${SERVER_ID} in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --server "$SERVER_ID" --num-servers "$NUM_SERVERS" --run --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# Validate eval result: check that #entries == #png files
# ============================================
needs_eval() {
    local dir=$1
    local eval_file="${dir}/categories_qwen3_vl_nudity.json"

    # No generation done
    if [ ! -f "${dir}/generation_stats.json" ]; then
        return 1  # skip, no images
    fi

    # No eval file → needs eval
    if [ ! -f "$eval_file" ]; then
        return 0
    fi

    # Count png files and json entries
    local num_images=$(ls "${dir}"/*.png 2>/dev/null | wc -l)
    local num_entries=$(/mnt/home/yhgil99/.conda/envs/vlm/bin/python -c "
import json, sys
try:
    with open('${eval_file}') as f:
        d = json.load(f)
    cats = [v['category'] for v in d.values()]
    valid = sum(1 for c in cats if c in ('Full','Partial','Safe','NotRel'))
    print(f'{len(d)} {valid}')
except:
    print('0 0')
" 2>/dev/null)
    local total_entries=$(echo $num_entries | awk '{print $1}')
    local valid_entries=$(echo $num_entries | awk '{print $2}')

    # Mismatch → delete and re-eval
    if [ "$num_images" -ne "$total_entries" ] || [ "$total_entries" -ne "$valid_entries" ]; then
        echo "  [INVALID] $(basename $dir): ${num_images} images, ${total_entries} entries, ${valid_entries} valid → re-eval"
        rm -f "$eval_file"
        return 0
    fi

    return 1  # eval is valid
}

# ============================================
# Find all dirs needing eval
# ============================================
declare -a ALL_DIRS=()
echo "Scanning for dirs needing eval..."
for dir in "${OUTPUT_BASE}"/mon*; do
    [ -d "$dir" ] || continue
    if needs_eval "$dir"; then
        ALL_DIRS+=("$dir")
    fi
done

TOTAL_ALL=${#ALL_DIRS[@]}

# Split for this server
CHUNK_SIZE=$(( (TOTAL_ALL + NUM_SERVERS - 1) / NUM_SERVERS ))
START=$(( SERVER_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if [ $END -gt $TOTAL_ALL ]; then END=$TOTAL_ALL; fi

declare -a MY_DIRS=()
for (( i=START; i<END; i++ )); do
    MY_DIRS+=("${ALL_DIRS[$i]}")
done

TOTAL=${#MY_DIRS[@]}

echo "=============================================="
echo "VLM EVAL V2 — SERVER ${SERVER_ID}/${NUM_SERVERS}"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Pending eval: ${TOTAL} / ${TOTAL_ALL} total"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Would evaluate:"
    for (( i=0; i<TOTAL; i++ )); do
        echo "  [${i}] GPU$((i % NUM_GPUS)) | $(basename ${MY_DIRS[$i]})"
    done
    echo ""
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

# ============================================
# Run eval
# ============================================
run_eval() {
    local GPU_IDX=$1
    local DIR=$2
    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local NAME=$(basename "$DIR")

    echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] EVAL: ${NAME}"
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} ${VLM_PYTHON} "${VLM_SCRIPT}" "${DIR}" nudity qwen 2>&1

    if [ $? -eq 0 ]; then
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] DONE: ${NAME}"
    else
        echo "[S${SERVER_ID} GPU${ACTUAL_GPU}] FAILED: ${NAME}"
    fi
}

TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS
IDX=0

for (( ci=0; ci<TOTAL; ci++ )); do
    DIR="${MY_DIRS[$ci]}"

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_eval $GPU_IDX "$DIR" &
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

# Summary
EVALUATED=$(find "${OUTPUT_BASE}" -name "categories_qwen3_vl_nudity.json" | wc -l)
echo ""
echo "=============================================="
echo "SERVER ${SERVER_ID} VLM EVAL COMPLETE"
echo "=============================================="
echo "Evaluated: ${EVALUATED} / ${TOTAL_ALL}"
echo "Done!"
