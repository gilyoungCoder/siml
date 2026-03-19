#!/bin/bash
# ============================================================================
# Batch Evaluation Script for Grid Search Results
# Evaluates all _skip folders using opensource VLM (Qwen2-VL)
#
# Usage:
#   ./batch_eval_grid_search.sh [concept] [model] [mode]
#   ./batch_eval_grid_search.sh all qwen background
#   ./batch_eval_grid_search.sh violence qwen foreground
# ============================================================================

set -u

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results"
LOG_DIR="/mnt/home/yhgil99/unlearning/vlm/logs"

# Parse arguments
CONCEPT=${1:-all}
MODEL=${2:-qwen}
MODE=${3:-background}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

# ============================================================================
# Concept to folder mapping
# ============================================================================
declare -A CONCEPT_FOLDERS
CONCEPT_FOLDERS["harassment"]="harassment_9class_step24300_skip"
CONCEPT_FOLDERS["hate"]="hate_9class_step20800_skip"
CONCEPT_FOLDERS["illegal"]="illegal_9class_step22600_skip"
CONCEPT_FOLDERS["selfharm"]="selfharm_9class_step20700_skip"
CONCEPT_FOLDERS["shocking"]="shocking_9class_step23700_skip"
CONCEPT_FOLDERS["violence"]="violence_9class_step15500_skip"

# I2P concept names for evaluation
declare -A EVAL_CONCEPTS
EVAL_CONCEPTS["harassment"]="harassment"
EVAL_CONCEPTS["hate"]="hate"
EVAL_CONCEPTS["illegal"]="illegal"
EVAL_CONCEPTS["selfharm"]="self_harm"
EVAL_CONCEPTS["shocking"]="shocking"
EVAL_CONCEPTS["violence"]="violence"

# ============================================================================
# Functions
# ============================================================================
print_header() {
    echo -e "\n${CYAN}============================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

count_images() {
    local dir=$1
    find "$dir" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l
}

evaluate_folder() {
    local concept_key=$1
    local folder_name=${CONCEPT_FOLDERS[$concept_key]}
    local eval_concept=${EVAL_CONCEPTS[$concept_key]}
    local folder_path="${BASE_DIR}/${folder_name}"

    if [ ! -d "$folder_path" ]; then
        echo -e "${RED}Folder not found: $folder_path${NC}"
        return 1
    fi

    print_header "Evaluating: $concept_key ($folder_name)"
    echo -e "Eval concept: $eval_concept"
    echo -e "Model: $MODEL"
    echo -e "Folder: $folder_path"

    # Count subfolders
    local num_subfolders=$(find "$folder_path" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo -e "Subfolders to evaluate: $num_subfolders"

    local log_file="${LOG_DIR}/eval_${concept_key}_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    echo -e "Log file: $log_file"

    if [ "$MODE" == "background" ]; then
        echo -e "${YELLOW}Running in BACKGROUND mode${NC}"

        nohup bash -c "
            cd /mnt/home/yhgil99/unlearning
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

            count=0
            total=\$(find '${folder_path}' -mindepth 1 -maxdepth 1 -type d | wc -l)

            for subfolder in ${folder_path}/*/; do
                if [ -d \"\$subfolder\" ]; then
                    count=\$((count + 1))
                    echo \"\"
                    echo \"=== [\$count/\$total] Processing: \$subfolder ===\"

                    # Check if already evaluated
                    if [ -f \"\${subfolder}results_${MODEL}_${eval_concept}.txt\" ]; then
                        echo \"Already evaluated, skipping...\"
                        continue
                    fi

                    python3 vlm/opensource_vlm_i2p_all.py \"\$subfolder\" ${eval_concept} ${MODEL}
                fi
            done

            echo \"\"
            echo \"=== COMPLETE: ${concept_key} ===\"
        " > "$log_file" 2>&1 &

        local pid=$!
        echo -e "${GREEN}Started with PID: $pid${NC}"
        echo "$pid" > "${LOG_DIR}/.eval_${concept_key}_${MODEL}.pid"
        echo -e "Monitor: tail -f $log_file"

    else
        echo -e "${YELLOW}Running in FOREGROUND mode${NC}"

        cd /mnt/home/yhgil99/unlearning

        local count=0
        local total=$(find "$folder_path" -mindepth 1 -maxdepth 1 -type d | wc -l)

        for subfolder in ${folder_path}/*/; do
            if [ -d "$subfolder" ]; then
                count=$((count + 1))
                echo ""
                echo "=== [$count/$total] Processing: $subfolder ==="

                # Check if already evaluated
                if [ -f "${subfolder}results_${MODEL}_${eval_concept}.txt" ]; then
                    echo "Already evaluated, skipping..."
                    continue
                fi

                python3 vlm/opensource_vlm_i2p_all.py "$subfolder" ${eval_concept} ${MODEL} 2>&1 | tee -a "$log_file"
            fi
        done
    fi
}

# ============================================================================
# Main
# ============================================================================
print_header "Grid Search Results Batch Evaluation"
echo -e "Concept: $CONCEPT"
echo -e "Model: $MODEL"
echo -e "Mode: $MODE"
echo -e "GPU: $CUDA_VISIBLE_DEVICES"
echo -e "Base dir: $BASE_DIR"

if [ "$CONCEPT" == "all" ]; then
    echo -e "\n${GREEN}Evaluating ALL concepts sequentially...${NC}"

    for concept_key in "${!CONCEPT_FOLDERS[@]}"; do
        evaluate_folder "$concept_key"

        if [ "$MODE" == "background" ]; then
            echo -e "${YELLOW}Waiting 5 seconds before next concept...${NC}"
            sleep 5
        fi
    done

    if [ "$MODE" == "background" ]; then
        echo ""
        print_header "All evaluations started in background"
        echo "Monitor with:"
        for concept_key in "${!CONCEPT_FOLDERS[@]}"; do
            echo "  tail -f ${LOG_DIR}/eval_${concept_key}_${MODEL}_*.log"
        done
    fi

else
    if [ -z "${CONCEPT_FOLDERS[$CONCEPT]+x}" ]; then
        echo -e "${RED}Unknown concept: $CONCEPT${NC}"
        echo "Available concepts: ${!CONCEPT_FOLDERS[@]}"
        exit 1
    fi

    evaluate_folder "$CONCEPT"
fi

echo ""
print_header "Batch Evaluation Launcher Complete"
