#!/bin/bash
# ============================================================================
# Parallel Batch Evaluation - Each concept on different GPU
#
# Usage:
#   ./batch_eval_grid_search_parallel.sh [model]
#   ./batch_eval_grid_search_parallel.sh qwen
#
# GPU Assignment:
#   GPU 0: harassment
#   GPU 1: hate
#   GPU 2: illegal
#   GPU 3: selfharm
#   GPU 4: shocking
#   GPU 5: violence
# ============================================================================

set -u

MODEL=${1:-qwen}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results"
LOG_DIR="/mnt/home/yhgil99/unlearning/vlm/logs"

mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}Parallel Grid Search Evaluation${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "Model: $MODEL"
echo -e "Log dir: $LOG_DIR"
echo ""

# Concept -> GPU mapping
declare -A GPU_MAP
GPU_MAP["harassment"]=6
GPU_MAP["hate"]=1
GPU_MAP["illegal"]=2
GPU_MAP["selfharm"]=3
GPU_MAP["shocking"]=4
GPU_MAP["violence"]=5

# Concept -> Folder mapping
declare -A CONCEPT_FOLDERS
CONCEPT_FOLDERS["harassment"]="harassment_9class_step24300_skip"
CONCEPT_FOLDERS["hate"]="hate_9class_step20800_skip"
CONCEPT_FOLDERS["illegal"]="illegal_9class_step22600_skip"
CONCEPT_FOLDERS["selfharm"]="selfharm_9class_step20700_skip"
CONCEPT_FOLDERS["shocking"]="shocking_9class_step23700_skip"
CONCEPT_FOLDERS["violence"]="violence_9class_step15500_skip"

# Concept -> Eval concept name
declare -A EVAL_CONCEPTS
EVAL_CONCEPTS["harassment"]="harassment"
EVAL_CONCEPTS["hate"]="hate"
EVAL_CONCEPTS["illegal"]="illegal"
EVAL_CONCEPTS["selfharm"]="self_harm"
EVAL_CONCEPTS["shocking"]="shocking"
EVAL_CONCEPTS["violence"]="violence"

# Launch each concept on its assigned GPU
for concept in harassment hate illegal selfharm shocking violence; do
    gpu=${GPU_MAP[$concept]}
    folder=${CONCEPT_FOLDERS[$concept]}
    eval_concept=${EVAL_CONCEPTS[$concept]}
    folder_path="${BASE_DIR}/${folder}"

    log_file="${LOG_DIR}/eval_${concept}_${MODEL}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${GREEN}Launching: ${concept}${NC}"
    echo -e "  GPU: $gpu"
    echo -e "  Folder: $folder"
    echo -e "  Log: $log_file"

    if [ ! -d "$folder_path" ]; then
        echo -e "  ${RED}Folder not found, skipping${NC}"
        continue
    fi

    num_subfolders=$(find "$folder_path" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo -e "  Subfolders: $num_subfolders"

    nohup bash -c "
        cd /mnt/home/yhgil99/unlearning
        export CUDA_VISIBLE_DEVICES=${gpu}

        count=0
        total=\$(find '${folder_path}' -mindepth 1 -maxdepth 1 -type d | wc -l)

        echo '============================================================'
        echo 'Concept: ${concept}'
        echo 'GPU: ${gpu}'
        echo 'Folder: ${folder_path}'
        echo 'Total subfolders: '\$total
        echo '============================================================'
        echo ''

        for subfolder in ${folder_path}/*/; do
            if [ -d \"\$subfolder\" ]; then
                count=\$((count + 1))
                echo ''
                echo \"=== [\$count/\$total] Processing: \$subfolder ===\"

                # Check if already evaluated
                if [ -f \"\${subfolder}results_qwen2_vl_${eval_concept}.txt\" ]; then
                    echo 'Already evaluated, skipping...'
                    continue
                fi

                python3 vlm/opensource_vlm_i2p_all.py \"\$subfolder\" ${eval_concept} ${MODEL}
            fi
        done

        echo ''
        echo '============================================================'
        echo 'COMPLETE: ${concept}'
        echo '============================================================'
    " > "$log_file" 2>&1 &

    pid=$!
    echo -e "  PID: $pid"
    echo "$pid" > "${LOG_DIR}/.eval_${concept}_${MODEL}.pid"
    echo ""

    # Small delay between launches to avoid race conditions
    sleep 2
done

echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}All 6 concepts launched on GPUs 0-5${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/eval_*_${MODEL}_*.log"
echo ""
echo "Or individual concepts:"
for concept in harassment hate illegal selfharm shocking violence; do
    echo "  tail -f ${LOG_DIR}/eval_${concept}_${MODEL}_*.log"
done
echo ""
echo "Check running processes:"
echo "  ps aux | grep opensource_vlm"
