#!/bin/bash
# ============================================================================
# Final Results Aggregation (per-method across datasets)
#
# Structure: final_{dataset}/{method}/results_qwen3_vl_nudity.txt
# Shows SR%, Safe, Partial, Full, NotRel per method per dataset.
#
# Usage:
#   bash aggregate_final_results.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${SCRIPT_DIR}/aggregated_final_results_${TIMESTAMP}.csv"

BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs"

DATASETS=(final_i2p final_mma final_p4dn final_ringabell final_unlearndiff)
METHODS=(sd_baseline safree esd ours_dual ours_mon3class ours_mon4class safree_dual safree_mon)

echo "============================================================"
echo "Final Results Aggregation (per-method across datasets)"
echo "============================================================"
echo ""

# CSV header
csv_header="method,dataset,sr,safe,partial,full,notrel,total"
echo "$csv_header" > "$OUTPUT_FILE"

# Per-dataset table
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  DATASET: ${ds}"
    echo "============================================================"
    printf "%-18s %6s %6s %6s %6s %6s %6s\n" "Method" "SR%" "Safe" "Part" "Full" "NRel" "Total"
    echo "--------------------------------------------------------------------"

    for method in "${METHODS[@]}"; do
        results_file="${BASE}/${ds}/${method}/results_qwen3_vl_nudity.txt"

        if [ -f "$results_file" ]; then
            sr=$(grep -oP 'SR.*\(\K[\d.]+' "$results_file" 2>/dev/null || echo "-")
            safe=$(grep -oP 'Safe: \K\d+' "$results_file" 2>/dev/null || echo "0")
            partial=$(grep -oP 'Partial: \K\d+' "$results_file" 2>/dev/null || echo "0")
            full=$(grep -oP 'Full: \K\d+' "$results_file" 2>/dev/null || echo "0")
            notrel=$(grep -oP 'NotRel: \K\d+' "$results_file" 2>/dev/null || echo "0")
            total=$(grep -oP 'Total images: \K\d+' "$results_file" 2>/dev/null || echo "0")

            printf "%-18s %6s %6s %6s %6s %6s %6s\n" "$method" "$sr" "$safe" "$partial" "$full" "$notrel" "$total"
            echo "${method},${ds},${sr},${safe},${partial},${full},${notrel},${total}" >> "$OUTPUT_FILE"
        else
            printf "%-18s %6s\n" "$method" "-"
        fi
    done
done

# Summary table: avg SR per method across datasets
echo ""
echo ""
echo "============================================================"
echo "  SUMMARY: Average SR% per method"
echo "============================================================"
printf "%-18s" "Method"
for ds in "${DATASETS[@]}"; do
    printf " %12s" "${ds#final_}"
done
printf " %10s\n" "Avg_SR"
echo "--------------------------------------------------------------------------------------------"

for method in "${METHODS[@]}"; do
    sr_sum=0
    count=0

    printf "%-18s" "$method"

    for ds in "${DATASETS[@]}"; do
        results_file="${BASE}/${ds}/${method}/results_qwen3_vl_nudity.txt"

        if [ -f "$results_file" ]; then
            sr=$(grep -oP 'SR.*\(\K[\d.]+' "$results_file" 2>/dev/null || echo "")
            if [ -n "$sr" ]; then
                printf " %12s" "$sr"
                sr_sum=$(awk "BEGIN {print $sr_sum + $sr}")
                count=$((count + 1))
            else
                printf " %12s" "-"
            fi
        else
            printf " %12s" "-"
        fi
    done

    if [ "$count" -gt 0 ]; then
        avg_sr=$(awk "BEGIN {printf \"%.1f\", $sr_sum / $count}")
    else
        avg_sr="-"
    fi
    printf " %10s\n" "$avg_sr"
done

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
