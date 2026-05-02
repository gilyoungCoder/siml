#!/bin/bash
# ============================================================================
# SD3 Experiments Master Script
# ============================================================================
# Runs all SD3 baselines across all datasets (nudity + I2P multi-concept + COCO)
#
# Usage:
#   ./scripts/sd3/run_sd3_all.sh                         # All methods, all datasets
#   ./scripts/sd3/run_sd3_all.sh --methods "baseline"     # Only vanilla baseline
#   ./scripts/sd3/run_sd3_all.sh --datasets "rab mma"     # Specific datasets only
#   ./scripts/sd3/run_sd3_all.sh --gpu 0                  # Single GPU
#
# Methods: baseline, safree, safe_denoiser, sgf
# Datasets: rab, mma, p4dn, unlearndiff, i2p_sexual, i2p_violence,
#           i2p_harassment, i2p_hate, i2p_shocking, i2p_illegal,
#           i2p_selfharm, coco
# ============================================================================

set -e

# ==== DEFAULTS ====
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE_DIR="/mnt/home3/yhgil99/unlearning"
SCRIPT_DIR="${BASE_DIR}/scripts/sd3"
OUTPUT_BASE="${BASE_DIR}/CAS_SpatialCFG/outputs/sd3"
LOG_DIR="${BASE_DIR}/logs/sd3"
PROMPT_DIR="${BASE_DIR}/SAFREE/datasets"

# SD3 generation defaults
STEPS=28
CFG=7.0
RESOLUTION=1024
SEED=42

# Methods and datasets
ALL_METHODS="baseline safree"
ALL_NUDITY_DATASETS="rab mma p4dn unlearndiff"
ALL_I2P_DATASETS="i2p_sexual i2p_violence i2p_harassment i2p_hate i2p_shocking i2p_illegal i2p_selfharm"
ALL_DATASETS="${ALL_NUDITY_DATASETS} ${ALL_I2P_DATASETS} coco"

# ==== PARSE ARGS ====
METHODS_ARG=""
DATASETS_ARG=""
GPU_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --methods)    METHODS_ARG="$2"; shift 2 ;;
        --datasets)   DATASETS_ARG="$2"; shift 2 ;;
        --gpu)        GPU_ARG="$2"; shift 2 ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --cfg)        CFG="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --output)     OUTPUT_BASE="$2"; shift 2 ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done

METHODS="${METHODS_ARG:-$ALL_METHODS}"
DATASETS="${DATASETS_ARG:-$ALL_DATASETS}"

# ==== DATASET → PROMPT FILE MAPPING ====
declare -A PROMPT_FILES=(
    # Nudity adversarial datasets
    ["rab"]="${PROMPT_DIR}/nudity-ring-a-bell.csv"
    ["mma"]="${PROMPT_DIR}/mma-diffusion-nsfw-adv-prompts.csv"
    ["p4dn"]="${PROMPT_DIR}/p4dn_16_prompt.csv"
    ["unlearndiff"]="${PROMPT_DIR}/unlearn_diff_nudity.csv"
    # I2P concept subsets
    ["i2p_sexual"]="${PROMPT_DIR}/i2p_categories/i2p_sexual.csv"
    ["i2p_violence"]="${PROMPT_DIR}/i2p_categories/i2p_violence.csv"
    ["i2p_harassment"]="${PROMPT_DIR}/i2p_categories/i2p_harassment.csv"
    ["i2p_hate"]="${PROMPT_DIR}/i2p_categories/i2p_hate.csv"
    ["i2p_shocking"]="${PROMPT_DIR}/i2p_categories/i2p_shocking.csv"
    ["i2p_illegal"]="${PROMPT_DIR}/i2p_categories/i2p_illegal_activity.csv"
    ["i2p_selfharm"]="${PROMPT_DIR}/i2p_categories/i2p_self-harm.csv"
    # Benign
    ["coco"]="${PROMPT_DIR}/coco_30k_10k.csv"
)

# Dataset → SAFREE concept mapping
declare -A DATASET_CONCEPT=(
    ["rab"]="sexual"
    ["mma"]="sexual"
    ["p4dn"]="sexual"
    ["unlearndiff"]="sexual"
    ["i2p_sexual"]="sexual"
    ["i2p_violence"]="violence"
    ["i2p_harassment"]="harassment"
    ["i2p_hate"]="hate"
    ["i2p_shocking"]="shocking"
    ["i2p_illegal"]="illegal"
    ["i2p_selfharm"]="selfharm"
    ["coco"]="none"
)

# ==== HELPER ====
find_free_gpu() {
    if [ -n "$GPU_ARG" ]; then
        echo "$GPU_ARG"
        return
    fi
    # Find GPU with least memory usage
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' '
}

mkdir -p "$LOG_DIR"

# ==== PRINT CONFIG ====
echo "=============================================="
echo "SD3 Experiments"
echo "=============================================="
echo "Methods:    ${METHODS}"
echo "Datasets:   ${DATASETS}"
echo "Steps:      ${STEPS}"
echo "CFG:        ${CFG}"
echo "Resolution: ${RESOLUTION}"
echo "Output:     ${OUTPUT_BASE}"
echo "=============================================="
echo ""

# ==== RUN ====
for method in $METHODS; do
    for dataset in $DATASETS; do
        prompt_file="${PROMPT_FILES[$dataset]}"
        concept="${DATASET_CONCEPT[$dataset]}"

        # Check prompt file exists
        if [ ! -f "$prompt_file" ]; then
            echo "[SKIP] $method/$dataset — prompt file not found: $prompt_file"
            continue
        fi

        outdir="${OUTPUT_BASE}/${method}/${dataset}"
        logfile="${LOG_DIR}/${method}_${dataset}.log"
        GPU=$(find_free_gpu)

        echo "[RUN] method=${method} dataset=${dataset} GPU=${GPU}"
        echo "      prompt=${prompt_file}"
        echo "      output=${outdir}"
        echo "      log=${logfile}"

        case "$method" in
            baseline)
                CUDA_VISIBLE_DEVICES=$GPU $PYTHON "${SCRIPT_DIR}/generate_sd3_baseline.py" \
                    --prompts "$prompt_file" \
                    --outdir "$outdir" \
                    --steps $STEPS --cfg_scale $CFG --resolution $RESOLUTION --seed $SEED \
                    > "$logfile" 2>&1
                ;;

            safree)
                CUDA_VISIBLE_DEVICES=$GPU $PYTHON "${SCRIPT_DIR}/generate_sd3_safree.py" \
                    --prompts "$prompt_file" \
                    --outdir "$outdir" \
                    --concept "$concept" \
                    --steps $STEPS --cfg_scale $CFG --resolution $RESOLUTION --seed $SEED \
                    > "$logfile" 2>&1
                ;;

            safe_denoiser)
                CUDA_VISIBLE_DEVICES=$GPU $PYTHON "${SCRIPT_DIR}/generate_sd3_safe_denoiser.py" \
                    --prompts "$prompt_file" \
                    --outdir "$outdir" \
                    --mode "safree_neg_prompt" \
                    --steps $STEPS --cfg_scale $CFG --resolution $RESOLUTION --seed $SEED \
                    > "$logfile" 2>&1
                ;;

            sgf)
                CUDA_VISIBLE_DEVICES=$GPU $PYTHON "${SCRIPT_DIR}/generate_sd3_sgf.py" \
                    --prompts "$prompt_file" \
                    --outdir "$outdir" \
                    --mode "sgf" \
                    --steps $STEPS --cfg_scale $CFG --resolution $RESOLUTION --seed $SEED \
                    > "$logfile" 2>&1
                ;;

            *)
                echo "[ERROR] Unknown method: $method"
                exit 1
                ;;
        esac

        status=$?
        if [ $status -eq 0 ]; then
            echo "      [DONE] ✓"
        else
            echo "      [FAIL] exit=$status — see $logfile"
        fi
        echo ""
    done
done

echo "=============================================="
echo "ALL COMPLETE"
echo "Output: ${OUTPUT_BASE}"
echo "Logs:   ${LOG_DIR}"
echo "=============================================="
