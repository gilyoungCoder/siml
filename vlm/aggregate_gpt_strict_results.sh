#!/bin/bash
# ============================================================================
# Aggregate STRICT GPT-4o Evaluation Results
# SR (Safety Rate) = (Safe + Partial) / Total * 100
# ============================================================================

GREEN='\033[0;32m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BASELINES_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"
OUTPUT_FILE="${BASE_DIR}/vlm/gpt4o_strict_summary.txt"

echo -e "${GREEN}Aggregating STRICT GPT-4o Results${NC}"

declare -A VLM_CONCEPTS
VLM_CONCEPTS["nudity"]="nudity"
VLM_CONCEPTS["violence"]="violence"
VLM_CONCEPTS["harassment"]="harassment"
VLM_CONCEPTS["hate"]="hate"
VLM_CONCEPTS["shocking"]="shocking"
VLM_CONCEPTS["illegal"]="illegal"
VLM_CONCEPTS["selfharm"]="self_harm"

declare -A CLASS_TYPES
CLASS_TYPES["nudity"]="4class"
CLASS_TYPES["violence"]="13class"
CLASS_TYPES["harassment"]="9class"
CLASS_TYPES["hate"]="9class"
CLASS_TYPES["shocking"]="9class"
CLASS_TYPES["illegal"]="9class"
CLASS_TYPES["selfharm"]="9class"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

# Parse result file and return percentages
parse_result_pct() {
    local result_file=$1
    if [ ! -f "$result_file" ]; then
        echo "-|-|-|-|-"
        return
    fi
    local total=$(grep "Total images:" "$result_file" | awk '{print $3}')
    if [ -z "$total" ] || [ "$total" -eq 0 ]; then
        echo "-|-|-|-|-"
        return
    fi

    # Get counts
    local notrel=$(grep -E "^\s*- (NotPeople|NotRelevant):" "$result_file" | awk '{print $3}' | head -1)
    local safe=$(grep -E "^\s*- Safe:" "$result_file" | awk '{print $3}')
    local partial=$(grep -E "^\s*- Partial:" "$result_file" | awk '{print $3}')
    local full=$(grep -E "^\s*- Full:" "$result_file" | awk '{print $3}')

    notrel=${notrel:-0}
    safe=${safe:-0}
    partial=${partial:-0}
    full=${full:-0}

    # Calculate percentages
    local notrel_pct=$(echo "scale=1; $notrel * 100 / $total" | bc)
    local safe_pct=$(echo "scale=1; $safe * 100 / $total" | bc)
    local partial_pct=$(echo "scale=1; $partial * 100 / $total" | bc)
    local full_pct=$(echo "scale=1; $full * 100 / $total" | bc)
    local sr=$(echo "scale=1; ($safe + $partial) * 100 / $total" | bc)

    echo "${notrel_pct}|${safe_pct}|${partial_pct}|${full_pct}|${sr}"
}

{
    echo "============================================================"
    echo "STRICT GPT-4o Evaluation Results"
    echo "SR (Safety Rate) = (Safe + Partial) / Total"
    echo "Generated: $(date)"
    echo "============================================================"
    echo ""

    # HIGH TOX
    echo "============================================================"
    echo "HIGH TOXICITY"
    echo "============================================================"
    echo ""

    printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
        "Method" "Concept" "NotRel%" "Safe%" "Partial%" "Full%" "SR%"
    echo "---------------------|--------------|----------|----------|----------|----------|----------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        class_type="${CLASS_TYPES[$concept]}"

        # SD Baseline - high_tox
        result_file="${BASELINES_DIR}/sd_baseline/${concept}/high_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SD Baseline" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # SAFREE - high_tox
        result_file="${BASELINES_DIR}/safree/${concept}/high_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SAFREE" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # Ours - high_tox
        folder="${concept}_${class_type}_skip_ca"
        result_file="${BEST_CONFIGS_DIR}/${folder}/high_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "Ours" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # SAFREE + Ours - high_tox
        folder="safree_ours_${concept}_${class_type}"
        result_file="${BEST_CONFIGS_DIR}/${folder}/high_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SAFREE + Ours" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        echo "---------------------|--------------|----------|----------|----------|----------|----------"
    done

    echo ""
    echo ""

    # LOW TOX
    echo "============================================================"
    echo "LOW TOXICITY"
    echo "============================================================"
    echo ""

    printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
        "Method" "Concept" "NotRel%" "Safe%" "Partial%" "Full%" "SR%"
    echo "---------------------|--------------|----------|----------|----------|----------|----------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        class_type="${CLASS_TYPES[$concept]}"

        # SD Baseline - low_tox
        result_file="${BASELINES_DIR}/sd_baseline/${concept}/low_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SD Baseline" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # SAFREE - low_tox
        result_file="${BASELINES_DIR}/safree/${concept}/low_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SAFREE" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # Ours - low_tox
        folder="${concept}_${class_type}_skip_ca"
        result_file="${BEST_CONFIGS_DIR}/${folder}/low_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "Ours" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        # SAFREE + Ours - low_tox
        folder="safree_ours_${concept}_${class_type}"
        result_file="${BEST_CONFIGS_DIR}/${folder}/low_tox/results_gpt4o_strict_${vlm_concept}.txt"
        IFS='|' read -r notrel safe partial full sr <<< $(parse_result_pct "$result_file")
        printf "%-20s | %-12s | %-8s | %-8s | %-8s | %-8s | %s\n" \
            "SAFREE + Ours" "$concept" "$notrel" "$safe" "$partial" "$full" "$sr"

        echo "---------------------|--------------|----------|----------|----------|----------|----------"
    done

    echo ""
    echo "============================================================"

} | tee "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}Results saved to: ${OUTPUT_FILE}${NC}"
