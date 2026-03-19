#!/bin/bash
# ============================================================================
# Generate MISSING mon4class configs (NO eval)
# ringabell 108개 config 기준, p4dn/unlearndiff 미완료분만 생성
# ============================================================================

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

USE_NOHUP=false
NUM_GPUS=8
SITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$SITE" ]; then
    echo "Usage: bash gen_missing_mon4class.sh --site A|B [--nohup]"
    exit 1
fi

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/gen_missing_mon4class_site${SITE}_${TIMESTAMP}.log"
    echo "Running in background (site $SITE)..."
    echo "Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" --site "$SITE" --num-gpus "$NUM_GPUS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_nudity_4class_sample_level_monitoring.py"

declare -A DATASET_PROMPTS
DATASET_PROMPTS[p4dn]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
DATASET_PROMPTS[unlearndiff]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/fine_grid_mon4class"

# ============================================
# COLLECT MISSING CONFIGS
# ============================================
# ringabell에 있는 config 목록에서 p4dn/unlearndiff에 없는 것만 수집
declare -a MISSING=()

for cfg_dir in "${OUTPUT_BASE}/ringabell"/mon*/; do
    [ -f "$cfg_dir/generation_stats.json" ] || continue
    cfg=$(basename "$cfg_dir")

    for ds in p4dn unlearndiff; do
        if [ ! -f "${OUTPUT_BASE}/${ds}/${cfg}/generation_stats.json" ]; then
            # parse config name: mon0.1_gs15_bs1.0_sp0.1-0.4
            mt=$(echo "$cfg" | sed 's/mon\([^_]*\)_.*/\1/')
            gs=$(echo "$cfg" | sed 's/.*_gs\([^_]*\)_.*/\1/')
            bs=$(echo "$cfg" | sed 's/.*_bs\([^_]*\)_.*/\1/')
            sp_start=$(echo "$cfg" | sed 's/.*_sp\([^-]*\)-.*/\1/')
            sp_end=$(echo "$cfg" | sed 's/.*-\(.*\)/\1/')
            MISSING+=("${ds}|${mt}|${gs}|${bs}|${sp_start}|${sp_end}")
        fi
    done
done

TOTAL=${#MISSING[@]}
echo "=============================================="
echo "MISSING MON4CLASS GENERATION (NO EVAL)"
echo "Total missing: $TOTAL"
echo "=============================================="

if [ $TOTAL -eq 0 ]; then
    echo "Nothing to generate!"
    exit 0
fi

# Split for sites
HALF=$(( (TOTAL + 1) / 2 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$HALF
elif [ "$SITE" = "B" ]; then
    START=$HALF; END=$TOTAL
else
    echo "Invalid site: $SITE"; exit 1
fi

echo "Site $SITE: $START to $((END-1)) (total $((END-START)))"

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
echo "GPUs: ${GPU_LIST[*]}"

# ============================================
# RUN
# ============================================
run_gen() {
    local GPU_IDX=$1 DS=$2 MT=$3 GS=$4 BS=$5 SP_START=$6 SP_END=$7
    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${DS}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${DS}_${EXP_NAME}.log"
    local PROMPT_FILE="${DATASET_PROMPTS[$DS]}"

    if [ -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP: ${DS}/${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] GEN: ${DS}/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
        --ckpt_path "${SD_CKPT}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --monitoring_threshold ${MT} \
        --guidance_scale ${GS} \
        --base_guidance_scale ${BS} \
        --spatial_threshold_start ${SP_START} \
        --spatial_threshold_end ${SP_END} \
        --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        >> "${LOG_FILE}" 2>&1

    if [ $? -ne 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${DS}/${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] DONE: ${DS}/${EXP_NAME}"
    fi
}

PIDS=()
GPU_IDX=0

for (( i=START; i<END; i++ )); do
    combo="${MISSING[$i]}"
    IFS='|' read -r DS MT GS BS SP_START SP_END <<< "$combo"

    run_gen $((GPU_IDX % ${#GPU_LIST[@]})) "$DS" "$MT" "$GS" "$BS" "$SP_START" "$SP_END" &
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
echo "=== Site $SITE generation complete ==="
