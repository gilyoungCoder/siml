#!/bin/bash
# ============================================================================
# Aggregate GPT-4o Evaluation Results for Best Configs
# ============================================================================
#
# Usage:
#   ./vlm/aggregate_best_configs_results.sh
#
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"
OUTPUT_FILE="${BASE_DIR}/vlm/best_configs_gpt4o_summary.txt"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Aggregating GPT-4o Results for Best Configs${NC}"
echo -e "${GREEN}============================================================${NC}"

# Concept mapping
declare -A CONFIGS
CONFIGS["nudity"]="nudity_4class_skip|nudity"
CONFIGS["violence"]="violence_13class_skip|violence"
CONFIGS["harassment"]="harassment_9class_skip|harassment"
CONFIGS["hate"]="hate_9class_skip|hate"
CONFIGS["shocking"]="shocking_9class_skip|shocking"
CONFIGS["illegal"]="illegal_9class_skip|illegal"
CONFIGS["selfharm"]="selfharm_9class_skip|self_harm"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

# Start output
{
    echo "============================================================"
    echo "GPT-4o Evaluation Results - Best Configs"
    echo "Generated: $(date)"
    echo "============================================================"
    echo ""

    # Table header
    printf "%-15s | %-10s | %-8s | %-8s | %-8s | %-8s | %s\n" \
        "Concept" "Tox Level" "Total" "Safe" "Partial" "Full" "Harmful Rate"
    echo "----------------|------------|----------|----------|----------|----------|-------------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        IFS='|' read -r folder_name vlm_concept <<< "${CONFIGS[$concept]}"

        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder_name}/${tox}/results_gpt4o_${vlm_concept}.txt"

            if [ -f "$result_file" ]; then
                # Parse the results file
                total=$(grep "Total images:" "$result_file" | awk '{print $3}')
                
                # Extract counts (handle both NotPeople and NotRelevant)
                safe=$(grep -E "^\s*- Safe:" "$result_file" | awk '{print $3}')
                partial=$(grep -E "^\s*- Partial:" "$result_file" | awk '{print $3}')
                full=$(grep -E "^\s*- Full:" "$result_file" | awk '{print $3}')
                
                # Get harmful rate
                harmful_line=$(grep "Harmful Rate" "$result_file")
                harmful_rate=$(echo "$harmful_line" | grep -oP '\(\K[0-9.]+(?=%)')
                
                if [ -n "$total" ]; then
                    printf "%-15s | %-10s | %-8s | %-8s | %-8s | %-8s | %s%%\n" \
                        "$concept" "$tox" "$total" "${safe:-0}" "${partial:-0}" "${full:-0}" "${harmful_rate:-N/A}"
                else
                    printf "%-15s | %-10s | %s\n" "$concept" "$tox" "Parse error"
                fi
            else
                printf "%-15s | %-10s | %s\n" "$concept" "$tox" "Not evaluated"
            fi
        done
    done

    echo ""
    echo "============================================================"
    echo "Summary by Concept (Combined high_tox + low_tox)"
    echo "============================================================"
    echo ""

    printf "%-15s | %-8s | %-8s | %-8s | %-8s | %s\n" \
        "Concept" "Total" "Safe" "Partial" "Full" "Harmful Rate"
    echo "----------------|----------|----------|----------|----------|-------------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        IFS='|' read -r folder_name vlm_concept <<< "${CONFIGS[$concept]}"

        total_all=0
        safe_all=0
        partial_all=0
        full_all=0

        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder_name}/${tox}/results_gpt4o_${vlm_concept}.txt"

            if [ -f "$result_file" ]; then
                t=$(grep "Total images:" "$result_file" | awk '{print $3}')
                s=$(grep -E "^\s*- Safe:" "$result_file" | awk '{print $3}')
                p=$(grep -E "^\s*- Partial:" "$result_file" | awk '{print $3}')
                f=$(grep -E "^\s*- Full:" "$result_file" | awk '{print $3}')

                total_all=$((total_all + ${t:-0}))
                safe_all=$((safe_all + ${s:-0}))
                partial_all=$((partial_all + ${p:-0}))
                full_all=$((full_all + ${f:-0}))
            fi
        done

        if [ $total_all -gt 0 ]; then
            harmful=$((partial_all + full_all))
            harmful_rate=$(echo "scale=1; $harmful * 100 / $total_all" | bc)
            printf "%-15s | %-8s | %-8s | %-8s | %-8s | %s%%\n" \
                "$concept" "$total_all" "$safe_all" "$partial_all" "$full_all" "$harmful_rate"
        else
            printf "%-15s | %s\n" "$concept" "No data"
        fi
    done

    echo ""
    echo "============================================================"

} | tee "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}Results saved to: ${OUTPUT_FILE}${NC}"
