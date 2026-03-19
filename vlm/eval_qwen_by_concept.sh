#!/bin/bash
# ============================================================================
# Qwen2-VL Evaluation by Concept
#
# Usage: bash eval_qwen_by_concept.sh <CONCEPT> <GPU>
# Example: bash eval_qwen_by_concept.sh nudity 0
#          bash eval_qwen_by_concept.sh harassment 1
#
# Concepts: nudity, harassment, hate, illegal, selfharm, shocking, violence
# ============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <CONCEPT> <GPU>"
    echo ""
    echo "Available concepts:"
    echo "  nudity1    - SAFREE 4class + SCG 4class + SCG 4class always"
    echo "  nudity2    - SCG 6class + SCG 6class skip"
    echo "  harassment - SAFREE 9class + SCG 9class"
    echo "  hate       - SAFREE 9class + SCG 9class"
    echo "  illegal    - SAFREE 9class + SCG 9class"
    echo "  selfharm   - SAFREE 9class + SCG 9class"
    echo "  shocking   - SAFREE 9class + SCG 9class"
    echo "  violence   - SAFREE 9class + SCG 9class"
    echo "  violence13 - SCG 13class + SCG 13class skip"
    echo ""
    echo "Example:"
    echo "  bash $0 nudity1 0"
    echo "  bash $0 nudity2 1"
    echo "  bash $0 violence13 2"
    exit 1
fi

CONCEPT=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=${GPU}

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"

# Map concept name for VLM evaluation
get_vlm_concept() {
    local c="$1"
    case "$c" in
        selfharm) echo "self_harm" ;;
        *) echo "$c" ;;
    esac
}

VLM_CONCEPT=$(get_vlm_concept "$CONCEPT")

# ============================================================================
# Helper function to check if folder needs evaluation
# ============================================================================
needs_evaluation() {
    local folder="$1"
    local concept="$2"

    local results_file="${folder}/results_qwen2_vl_${concept}.txt"
    if [ -f "$results_file" ]; then
        return 1
    fi

    local img_count=$(ls "$folder"/*.png 2>/dev/null | wc -l)
    if [ "$img_count" -eq 0 ]; then
        return 1
    fi

    return 0
}

# ============================================================================
# Evaluate function
# ============================================================================
evaluate_folder() {
    local folder="$1"
    local concept="$2"
    local folder_name=$(basename "$folder")

    if needs_evaluation "$folder" "$concept"; then
        echo "[EVAL] $folder_name"
        python "$VLM_SCRIPT" "$folder" "$concept" qwen
    else
        echo "[SKIP] $folder_name"
    fi
}

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "Qwen2-VL Evaluation for: ${CONCEPT}"
echo "GPU: ${GPU}"
echo "VLM Concept: ${VLM_CONCEPT}"
echo "=============================================="

total_evaluated=0
total_skipped=0

case "$CONCEPT" in
    nudity1)
        VLM_CONCEPT="nudity"
        echo ""
        echo ">>> SAFREE 4-class nudity"
        SAFREE_4CLASS="/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_4class_nudity"
        if [ -d "$SAFREE_4CLASS" ]; then
            for folder in "$SAFREE_4CLASS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi

        echo ""
        echo ">>> SCG nudity 4-class"
        SCG_4CLASS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class"
        if [ -d "$SCG_4CLASS" ]; then
            for folder in "$SCG_4CLASS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi

        echo ""
        echo ">>> SCG nudity 4-class always"
        SCG_4CLASS_ALWAYS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results/nudity_4class_always"
        if [ -d "$SCG_4CLASS_ALWAYS" ]; then
            for folder in "$SCG_4CLASS_ALWAYS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi
        ;;

    nudity2)
        VLM_CONCEPT="nudity"
        echo ""
        echo ">>> SCG nudity 6-class v2 step22700"
        SCG_6CLASS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700"
        if [ -d "$SCG_6CLASS" ]; then
            for folder in "$SCG_6CLASS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi

        echo ""
        echo ">>> SCG nudity 6-class v2 step22700 skip"
        SCG_6CLASS_SKIP="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip"
        if [ -d "$SCG_6CLASS_SKIP" ]; then
            for folder in "$SCG_6CLASS_SKIP"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi
        ;;

    harassment|hate|illegal|selfharm|shocking|violence)
        # Get step number
        declare -A STEP_MAP
        STEP_MAP["harassment"]="24300"
        STEP_MAP["hate"]="20800"
        STEP_MAP["illegal"]="22600"
        STEP_MAP["selfharm"]="20700"
        STEP_MAP["shocking"]="23700"
        STEP_MAP["violence"]="15500"
        STEP=${STEP_MAP[$CONCEPT]}

        echo ""
        echo ">>> SAFREE 9-class ${CONCEPT}"
        SAFREE_9CLASS="/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_9class/${CONCEPT}_step${STEP}"
        if [ -d "$SAFREE_9CLASS" ]; then
            for folder in "$SAFREE_9CLASS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi

        echo ""
        echo ">>> SCG 9-class ${CONCEPT}"
        SCG_9CLASS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results/${CONCEPT}_9class_step${STEP}"
        if [ -d "$SCG_9CLASS" ]; then
            for folder in "$SCG_9CLASS"/*/; do
                [ -d "$folder" ] || continue
                evaluate_folder "$folder" "$VLM_CONCEPT"
            done
        fi

        # SCG 9-class skip folders are already evaluated, skipping
        echo ""
        echo ">>> SCG 9-class ${CONCEPT} skip - Already evaluated, skipping"
        ;;

    *)
        echo "Unknown concept: $CONCEPT"
        echo "Available: nudity, harassment, hate, illegal, selfharm, shocking, violence"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Evaluation Complete for: ${CONCEPT}"
echo "=============================================="
