#!/bin/bash
# ============================================================================
# Aggregate MODIFIED GPT-4o Evaluation Results
# (For results from gpt_i2p_all_modified.py)
# ============================================================================
#
# Usage:
#   ./vlm/aggregate_gpt_modified_results.sh
#
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/mnt/home/yhgil99/unlearning"
BASELINES_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"
BEST_CONFIGS_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"
OUTPUT_FILE="${BASE_DIR}/vlm/gpt4o_modified_summary.txt"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Aggregating MODIFIED GPT-4o Results${NC}"
echo -e "${GREEN}============================================================${NC}"

# VLM concept mapping
declare -A VLM_CONCEPTS
VLM_CONCEPTS["nudity"]="nudity"
VLM_CONCEPTS["violence"]="violence"
VLM_CONCEPTS["harassment"]="harassment"
VLM_CONCEPTS["hate"]="hate"
VLM_CONCEPTS["shocking"]="shocking"
VLM_CONCEPTS["illegal"]="illegal"
VLM_CONCEPTS["selfharm"]="self_harm"

# Class type mapping
declare -A CLASS_TYPES
CLASS_TYPES["nudity"]="4class"
CLASS_TYPES["violence"]="13class"
CLASS_TYPES["harassment"]="9class"
CLASS_TYPES["hate"]="9class"
CLASS_TYPES["shocking"]="9class"
CLASS_TYPES["illegal"]="9class"
CLASS_TYPES["selfharm"]="9class"

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

# Function to parse result file and extract stats
parse_result() {
    local result_file=$1

    if [ ! -f "$result_file" ]; then
        echo "N/A|N/A|N/A|N/A|N/A"
        return
    fi

    local total=$(grep "Total images:" "$result_file" | awk '{print $3}')
    local safe=$(grep -E "^\s*- Safe:" "$result_file" | awk '{print $3}')
    local partial=$(grep -E "^\s*- Partial:" "$result_file" | awk '{print $3}')
    local full=$(grep -E "^\s*- Full:" "$result_file" | awk '{print $3}')
    local harmful_rate=$(grep "Harmful Rate" "$result_file" | grep -oP '\(\K[0-9.]+(?=%)' || echo "N/A")

    echo "${total:-0}|${safe:-0}|${partial:-0}|${full:-0}|${harmful_rate}"
}

# Start output
{
    echo "============================================================"
    echo "MODIFIED GPT-4o Evaluation Results"
    echo "(Using lenient prompts from gpt_i2p_all_modified.py)"
    echo "Generated: $(date)"
    echo "============================================================"
    echo ""

    # ========================================================================
    # Section 1: Baselines (SD Baseline + SAFREE)
    # ========================================================================
    echo "============================================================"
    echo "SECTION 1: BASELINES_I2P"
    echo "============================================================"
    echo ""

    printf "%-12s | %-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s\n" \
        "Method" "Concept" "Tox" "Total" "Safe" "Partial" "Full" "Harmful%"
    echo "-------------|--------------|----------|--------|--------|---------|-------|----------"

    for method in "sd_baseline" "safree"; do
        for concept in "${ALL_CONCEPTS[@]}"; do
            vlm_concept="${VLM_CONCEPTS[$concept]}"

            for tox in "high_tox" "low_tox"; do
                result_file="${BASELINES_DIR}/${method}/${concept}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
                IFS='|' read -r total safe partial full harmful <<< $(parse_result "$result_file")

                if [ "$total" != "N/A" ] && [ "$total" != "0" ]; then
                    printf "%-12s | %-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                        "$method" "$concept" "$tox" "$total" "$safe" "$partial" "$full" "$harmful"
                else
                    printf "%-12s | %-12s | %-8s | %s\n" "$method" "$concept" "$tox" "Not evaluated"
                fi
            done
        done
        echo "-------------|--------------|----------|--------|--------|---------|-------|----------"
    done

    echo ""

    # ========================================================================
    # Section 2: Best Configs (Ours skip_ca)
    # ========================================================================
    echo "============================================================"
    echo "SECTION 2: BEST_CONFIGS (Ours - skip_ca)"
    echo "============================================================"
    echo ""

    printf "%-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s\n" \
        "Concept" "Tox" "Total" "Safe" "Partial" "Full" "Harmful%"
    echo "-------------|----------|--------|--------|---------|-------|----------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        class_type="${CLASS_TYPES[$concept]}"
        folder="${concept}_${class_type}_skip_ca"

        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
            IFS='|' read -r total safe partial full harmful <<< $(parse_result "$result_file")

            if [ "$total" != "N/A" ] && [ "$total" != "0" ]; then
                printf "%-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                    "$concept" "$tox" "$total" "$safe" "$partial" "$full" "$harmful"
            else
                printf "%-12s | %-8s | %s\n" "$concept" "$tox" "Not evaluated"
            fi
        done
    done

    echo ""

    # ========================================================================
    # Section 3: Best Configs (SAFREE + Ours)
    # ========================================================================
    echo "============================================================"
    echo "SECTION 3: BEST_CONFIGS (SAFREE + Ours)"
    echo "============================================================"
    echo ""

    printf "%-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s\n" \
        "Concept" "Tox" "Total" "Safe" "Partial" "Full" "Harmful%"
    echo "-------------|----------|--------|--------|---------|-------|----------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        class_type="${CLASS_TYPES[$concept]}"
        folder="safree_ours_${concept}_${class_type}"

        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
            IFS='|' read -r total safe partial full harmful <<< $(parse_result "$result_file")

            if [ "$total" != "N/A" ] && [ "$total" != "0" ]; then
                printf "%-12s | %-8s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                    "$concept" "$tox" "$total" "$safe" "$partial" "$full" "$harmful"
            else
                printf "%-12s | %-8s | %s\n" "$concept" "$tox" "Not evaluated"
            fi
        done
    done

    echo ""

    # ========================================================================
    # Section 4: Comparison Summary (Combined high_tox + low_tox)
    # ========================================================================
    echo "============================================================"
    echo "SECTION 4: COMPARISON SUMMARY (Combined high+low tox)"
    echo "============================================================"
    echo ""

    printf "%-20s | %-12s | %-6s | %-6s | %-7s | %-5s | %s\n" \
        "Method" "Concept" "Total" "Safe" "Partial" "Full" "Harmful%"
    echo "---------------------|--------------|--------|--------|---------|-------|----------"

    for concept in "${ALL_CONCEPTS[@]}"; do
        vlm_concept="${VLM_CONCEPTS[$concept]}"
        class_type="${CLASS_TYPES[$concept]}"

        # SD Baseline
        total_all=0; safe_all=0; partial_all=0; full_all=0
        for tox in "high_tox" "low_tox"; do
            result_file="${BASELINES_DIR}/sd_baseline/${concept}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
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
            printf "%-20s | %-12s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                "SD Baseline" "$concept" "$total_all" "$safe_all" "$partial_all" "$full_all" "$harmful_rate"
        fi

        # SAFREE Baseline
        total_all=0; safe_all=0; partial_all=0; full_all=0
        for tox in "high_tox" "low_tox"; do
            result_file="${BASELINES_DIR}/safree/${concept}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
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
            printf "%-20s | %-12s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                "SAFREE" "$concept" "$total_all" "$safe_all" "$partial_all" "$full_all" "$harmful_rate"
        fi

        # Ours (skip_ca)
        folder="${concept}_${class_type}_skip_ca"
        total_all=0; safe_all=0; partial_all=0; full_all=0
        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
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
            printf "%-20s | %-12s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                "Ours (skip_ca)" "$concept" "$total_all" "$safe_all" "$partial_all" "$full_all" "$harmful_rate"
        fi

        # SAFREE + Ours
        folder="safree_ours_${concept}_${class_type}"
        total_all=0; safe_all=0; partial_all=0; full_all=0
        for tox in "high_tox" "low_tox"; do
            result_file="${BEST_CONFIGS_DIR}/${folder}/${tox}/results_gpt4o_mod_${vlm_concept}.txt"
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
            printf "%-20s | %-12s | %-6s | %-6s | %-7s | %-5s | %s%%\n" \
                "SAFREE + Ours" "$concept" "$total_all" "$safe_all" "$partial_all" "$full_all" "$harmful_rate"
        fi

        echo "---------------------|--------------|--------|--------|---------|-------|----------"
    done

    echo ""
    echo "============================================================"

} | tee "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}Results saved to: ${OUTPUT_FILE}${NC}"
