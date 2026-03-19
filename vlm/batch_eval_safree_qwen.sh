#!/bin/bash
# ============================================================================
# Batch Qwen2-VL Evaluation for SAFREE++ and SoftDelete+CG Grid Search Results
#
# Usage: bash batch_eval_safree_qwen.sh <GPU>
# Example: bash batch_eval_safree_qwen.sh 0
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <GPU>"
    echo "Example: $0 0"
    exit 1
fi

GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"

# ============================================================================
# Helper function to determine concept from folder path
# ============================================================================
get_concept() {
    local path="$1"

    # Check for nudity/sexual
    if [[ "$path" == *"nudity"* ]] || [[ "$path" == *"sexual"* ]]; then
        echo "nudity"
    elif [[ "$path" == *"violence"* ]]; then
        echo "violence"
    elif [[ "$path" == *"harassment"* ]]; then
        echo "harassment"
    elif [[ "$path" == *"hate"* ]]; then
        echo "hate"
    elif [[ "$path" == *"shocking"* ]]; then
        echo "shocking"
    elif [[ "$path" == *"illegal"* ]]; then
        echo "illegal"
    elif [[ "$path" == *"selfharm"* ]] || [[ "$path" == *"self_harm"* ]] || [[ "$path" == *"self-harm"* ]]; then
        echo "self_harm"
    else
        echo "unknown"
    fi
}

# ============================================================================
# Helper function to check if folder needs evaluation
# ============================================================================
needs_evaluation() {
    local folder="$1"
    local concept="$2"

    # Check if results file already exists
    local results_file="${folder}/results_qwen2_vl_${concept}.txt"
    if [ -f "$results_file" ]; then
        return 1  # false - doesn't need evaluation
    fi

    # Check if folder contains images
    local img_count=$(ls "$folder"/*.png 2>/dev/null | wc -l)
    if [ "$img_count" -eq 0 ]; then
        return 1  # false - no images
    fi

    return 0  # true - needs evaluation
}

# ============================================================================
# Evaluate all subfolders in a directory
# ============================================================================
evaluate_directory() {
    local base_dir="$1"
    local dir_name=$(basename "$base_dir")

    echo ""
    echo "=============================================="
    echo "Scanning: $base_dir"
    echo "=============================================="

    # Skip if directory doesn't exist
    if [ ! -d "$base_dir" ]; then
        echo "Directory not found: $base_dir"
        return
    fi

    # Find all subfolders with images
    for subfolder in "$base_dir"/*/; do
        [ -d "$subfolder" ] || continue

        local folder_name=$(basename "$subfolder")
        local concept=$(get_concept "$subfolder")

        if [ "$concept" == "unknown" ]; then
            # Try to get concept from parent directory name
            concept=$(get_concept "$base_dir")
        fi

        if [ "$concept" == "unknown" ]; then
            echo "[SKIP] Unknown concept: $subfolder"
            continue
        fi

        if needs_evaluation "$subfolder" "$concept"; then
            echo ""
            echo "[EVAL] $folder_name (concept: $concept)"
            python "$VLM_SCRIPT" "$subfolder" "$concept" qwen
        else
            echo "[SKIP] Already evaluated or no images: $folder_name"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "Batch Qwen2-VL Evaluation"
echo "=============================================="
echo "GPU: ${GPU}"
echo "=============================================="

# 1. SAFREE 9-class grid search results
SAFREE_9CLASS_BASE="/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_9class"
if [ -d "$SAFREE_9CLASS_BASE" ]; then
    for concept_dir in "$SAFREE_9CLASS_BASE"/*/; do
        [ -d "$concept_dir" ] || continue
        evaluate_directory "$concept_dir"
    done
fi

# 2. SAFREE 4-class nudity grid search results
SAFREE_4CLASS_NUDITY="/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_4class_nudity"
if [ -d "$SAFREE_4CLASS_NUDITY" ]; then
    echo ""
    echo "=============================================="
    echo "Evaluating: SAFREE 4-class nudity"
    echo "=============================================="
    for subfolder in "$SAFREE_4CLASS_NUDITY"/*/; do
        [ -d "$subfolder" ] || continue
        folder_name=$(basename "$subfolder")

        if needs_evaluation "$subfolder" "nudity"; then
            echo ""
            echo "[EVAL] $folder_name (concept: nudity)"
            python "$VLM_SCRIPT" "$subfolder" nudity qwen
        else
            echo "[SKIP] Already evaluated or no images: $folder_name"
        fi
    done
fi

# 3. SoftDelete+CG nudity 6-class v2 step22700
NUDITY_6CLASS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700"
if [ -d "$NUDITY_6CLASS" ]; then
    echo ""
    echo "=============================================="
    echo "Evaluating: Nudity 6-class v2 step22700"
    echo "=============================================="
    for subfolder in "$NUDITY_6CLASS"/*/; do
        [ -d "$subfolder" ] || continue
        folder_name=$(basename "$subfolder")

        if needs_evaluation "$subfolder" "nudity"; then
            echo ""
            echo "[EVAL] $folder_name (concept: nudity)"
            python "$VLM_SCRIPT" "$subfolder" nudity qwen
        else
            echo "[SKIP] Already evaluated or no images: $folder_name"
        fi
    done
fi

# 4. SoftDelete+CG nudity 6-class v2 step22700 skip
NUDITY_6CLASS_SKIP="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_nudity_6class_v2_step22700_skip"
if [ -d "$NUDITY_6CLASS_SKIP" ]; then
    echo ""
    echo "=============================================="
    echo "Evaluating: Nudity 6-class v2 step22700 skip"
    echo "=============================================="
    for subfolder in "$NUDITY_6CLASS_SKIP"/*/; do
        [ -d "$subfolder" ] || continue
        folder_name=$(basename "$subfolder")

        if needs_evaluation "$subfolder" "nudity"; then
            echo ""
            echo "[EVAL] $folder_name (concept: nudity)"
            python "$VLM_SCRIPT" "$subfolder" nudity qwen
        else
            echo "[SKIP] Already evaluated or no images: $folder_name"
        fi
    done
fi

# 5. SoftDelete+CG grid search results (all concepts, excluding _skip folders per user request)
# Wait, user said skip folders already done, not to skip _skip folders. Let me evaluate all.
SCG_GRID_RESULTS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_results"
if [ -d "$SCG_GRID_RESULTS" ]; then
    for concept_dir in "$SCG_GRID_RESULTS"/*/; do
        [ -d "$concept_dir" ] || continue
        evaluate_directory "$concept_dir"
    done
fi

echo ""
echo "=============================================="
echo "Batch Evaluation Complete!"
echo "=============================================="
