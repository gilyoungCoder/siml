#!/bin/bash
# ============================================================================
# Grid Search: SAFREE + Monitoring (Multi-Dataset, Focused)
#
# Based on best ringabell results:
#   mon0.3_gs12.5_bs2.0_sp0.5-0.5 (89.9%)
#   mon0.3_gs7.5_bs0.0_sp0.3-0.3  (89.9%, less guidance)
#
# Usage:
#   ./grid_search_safree_mon_multi_dataset.sh                  # Run all
#   ./grid_search_safree_mon_multi_dataset.sh --dry-run        # Print only
#   ./grid_search_safree_mon_multi_dataset.sh --nohup          # Background
#   ./grid_search_safree_mon_multi_dataset.sh --num-gpus 4
#   ./grid_search_safree_mon_multi_dataset.sh --datasets "p4dn ringabell"
# ============================================================================

set -e

DRY_RUN=false
USE_NOHUP=false
NUM_GPUS=16
CONCEPT="nudity"
DATASETS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --nohup) USE_NOHUP=true; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --concept) CONCEPT="$2"; shift 2 ;;
        --datasets) DATASETS_ARG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./results/grid_search_safree_mon_multi_${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/grid_search.log"
    echo "Running in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    if [ -n "$DATASETS_ARG" ]; then
        nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --concept "${CONCEPT}" --num-gpus "${NUM_GPUS}" --datasets "${DATASETS_ARG}" > "${LOG_FILE}" 2>&1 &
    else
        nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --concept "${CONCEPT}" --num-gpus "${NUM_GPUS}" > "${LOG_FILE}" 2>&1 &
    fi
    echo "Background PID: $!"
    exit 0
fi

cd /mnt/home/yhgil99/unlearning/SAFREE

# ============================================
# DATASET CONFIGURATION
# ============================================

declare -A DATASET_PROMPTS
DATASET_PROMPTS=(
    ["p4dn"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
    ["ringabell"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
    ["unlearndiff"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
)

if [ -n "$DATASETS_ARG" ]; then
    IFS=' ' read -ra DATASETS <<< "$DATASETS_ARG"
else
    DATASETS=(p4dn ringabell unlearndiff)
fi

for ds in "${DATASETS[@]}"; do
    if [ -z "${DATASET_PROMPTS[$ds]:-}" ]; then
        echo "ERROR: Unknown dataset: $ds"; exit 1
    fi
    if [ ! -f "${DATASET_PROMPTS[$ds]}" ]; then
        echo "ERROR: Prompt file not found: ${DATASET_PROMPTS[$ds]}"; exit 1
    fi
done

# ============================================
# FIXED CONFIG
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5

SAFREE_ALPHA=0.01
SVF_UP_T=10
CATEGORY="nudity"
THRESHOLD_STRATEGY="cosine"
HARMFUL_SCALE=1.0

# ============================================
# GRID SEARCH PARAMETERS (Focused)
# ============================================

# Monitoring: 0.3 was best, try nearby
MONITORING_THRESHOLDS=(0.2 0.3 0.4)

# Guidance: 7.5 (light) and 12.5 (best) + middle
GUIDANCE_SCALES=(5 7.5 10.0)

# Base: 0.0 (light) and 2.0 (best)
BASE_SCALES=(0.0 1.0 2.0)

# Spatial: focus on best combos
SPATIAL_THRESHOLDS=(
    "0.3 0.3"
    "0.5 0.5"
    "0.7 0.3"
)

# Total: 3 datasets × 3 mon × 3 gs × 3 bs × 4 sp = 324 experiments

# ============================================
# SETUP
# ============================================

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
NUM_GPUS=${#GPU_LIST[@]}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./results/grid_search_safree_mon_multi_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}/logs"

echo "=============================================="
echo "SAFREE + MONITORING GRID SEARCH (Multi-Dataset)"
echo "=============================================="
echo "Datasets: ${DATASETS[*]}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

# Build combinations
declare -a COMBINATIONS=()
for ds in "${DATASETS[@]}"; do
    for mt in "${MONITORING_THRESHOLDS[@]}"; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SPATIAL_THRESHOLDS[@]}"; do
                    COMBINATIONS+=("${ds}|${mt}|${gs}|${bs}|${sp}")
                done
            done
        done
    done
done

TOTAL=${#COMBINATIONS[@]}
echo "Total combinations: ${TOTAL}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE]"
    for combo in "${COMBINATIONS[@]}"; do
        IFS='|' read -r DS MT GS BS SP <<< "$combo"
        SP_START=$(echo $SP | awk '{print $1}')
        SP_END=$(echo $SP | awk '{print $2}')
        echo "  ${DS}/mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    done
    echo ""
    echo "Total: ${TOTAL} experiments on ${NUM_GPUS} GPUs"
    exit 0
fi

# Save config
cat > "${OUTPUT_BASE}/config.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "datasets": $(printf '%s\n' "${DATASETS[@]}" | jq -R . | jq -s .),
    "total_experiments": ${TOTAL},
    "num_gpus": ${NUM_GPUS},
    "parameters": {
        "monitoring_thresholds": [$(IFS=,; echo "${MONITORING_THRESHOLDS[*]}")],
        "guidance_scales": [$(IFS=,; echo "${GUIDANCE_SCALES[*]}")],
        "base_scales": [$(IFS=,; echo "${BASE_SCALES[*]}")],
        "harmful_scale": ${HARMFUL_SCALE}
    }
}
EOF

# ============================================
# RUN
# ============================================

run_experiment() {
    local GPU_IDX=$1
    local DATASET=$2
    local MON_THR=$3
    local GUIDANCE_SCALE=$4
    local BASE_SCALE=$5
    local SPATIAL_START=$6
    local SPATIAL_END=$7
    local EXP_NAME=$8

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local OUTPUT_DIR="${OUTPUT_BASE}/${DATASET}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${DATASET}_${EXP_NAME}.log"
    local PROMPT_FILE="${DATASET_PROMPTS[$DATASET]}"

    echo "[GPU ${ACTUAL_GPU}] Starting: ${DATASET}/${EXP_NAME}"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python generate_safree_monitoring.py \
        --ckpt_path "${SD_CKPT}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --safree \
        --safree_alpha ${SAFREE_ALPHA} \
        --svf \
        --svf_up_t ${SVF_UP_T} \
        --category "${CATEGORY}" \
        --monitoring_threshold ${MON_THR} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --harmful_scale ${HARMFUL_SCALE} \
        --base_guidance_scale ${BASE_SCALE} \
        --spatial_threshold_start ${SPATIAL_START} \
        --spatial_threshold_end ${SPATIAL_END} \
        --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        > "${LOG_FILE}" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] Completed: ${DATASET}/${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${DATASET}/${EXP_NAME} (exit: ${STATUS})"
    fi
    return $STATUS
}

declare -A RUNNING_JOBS

IDX=0
for combo in "${COMBINATIONS[@]}"; do
    IFS='|' read -r DS MT GS BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')
    EXP_NAME="mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"

    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            JOB_PID="${RUNNING_JOBS[$gpu]:-}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                run_experiment $gpu "$DS" $MT $GS $BS $SP_START $SP_END "$EXP_NAME" &
                RUNNING_JOBS[$gpu]=$!
                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/${TOTAL}"
                sleep 2
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
echo "GRID SEARCH COMPLETE"
echo "=============================================="
echo "Results saved to: ${OUTPUT_BASE}"
echo ""

# Summary
echo "Generating summary..."
python -c "
import json
from pathlib import Path

base_dir = Path('${OUTPUT_BASE}')
all_results = {}

for ds_dir in sorted(base_dir.iterdir()):
    if not ds_dir.is_dir() or ds_dir.name == 'logs':
        continue
    ds_name = ds_dir.name
    results = []
    for exp_dir in sorted(ds_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        stats_file = exp_dir / 'generation_stats.json'
        if not stats_file.exists():
            continue
        with open(stats_file) as f:
            data = json.load(f)
        args = data.get('args', {})
        overall = data.get('overall', {})
        results.append({
            'experiment': exp_dir.name,
            'monitoring_threshold': args.get('monitoring_threshold'),
            'guidance_scale': args.get('guidance_scale'),
            'base_guidance_scale': args.get('base_guidance_scale'),
            'spatial_start': args.get('spatial_threshold_start'),
            'spatial_end': args.get('spatial_threshold_end'),
            'avg_guided_steps': overall.get('avg_guided_steps', 0),
            'avg_guidance_ratio': overall.get('avg_guidance_ratio', 0),
        })
    results.sort(key=lambda x: x['avg_guidance_ratio'], reverse=True)
    all_results[ds_name] = results

summary_file = base_dir / 'grid_search_summary.json'
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'Summary saved to: {summary_file}')
for ds, res in all_results.items():
    print(f'  {ds}: {len(res)} experiments')
"
echo "Done!"
