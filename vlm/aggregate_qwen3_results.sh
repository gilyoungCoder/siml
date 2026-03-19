#!/bin/bash
# ============================================================================
# Aggregate Qwen3-VL Results
# Parses results_qwen3_vl_nudity.txt from all subdirectories and creates summary
#
# Usage:
#   ./aggregate_qwen3_results.sh <dir1> [dir2] ...
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${SCRIPT_DIR}/aggregated_qwen3_results_${TIMESTAMP}.csv"

echo "============================================================"
echo "Aggregating Qwen3-VL Results"
echo "============================================================"
echo ""

# CSV header
echo "directory,experiment,safe,partial,full,notrel,total,sr_percent" > "$OUTPUT_FILE"

# Collect all results
declare -a RESULTS=()

for base_dir in "$@"; do
    if [ ! -d "$base_dir" ]; then
        echo "Warning: Directory not found: $base_dir"
        continue
    fi

    base_name=$(basename "$base_dir")
    echo "Scanning: $base_name"

    # Check if base_dir itself has results
    if [ -f "$base_dir/results_qwen3_vl_nudity.txt" ]; then
        results_file="$base_dir/results_qwen3_vl_nudity.txt"

        safe=$(grep -oP 'Safe: \K\d+' "$results_file" 2>/dev/null || echo "0")
        partial=$(grep -oP 'Partial: \K\d+' "$results_file" 2>/dev/null || echo "0")
        full=$(grep -oP 'Full: \K\d+' "$results_file" 2>/dev/null || echo "0")
        notrel=$(grep -oP 'NotRel: \K\d+' "$results_file" 2>/dev/null || echo "0")
        total=$(grep -oP 'Total images: \K\d+' "$results_file" 2>/dev/null || echo "0")
        sr=$(grep -oP 'SR.*\(\K[\d.]+' "$results_file" 2>/dev/null || echo "0")

        echo "$base_name,ROOT,$safe,$partial,$full,$notrel,$total,$sr" >> "$OUTPUT_FILE"
        RESULTS+=("$sr|$base_name|ROOT|$safe|$partial|$full|$notrel|$total")
    fi

    # Check subdirectories
    for subdir in "$base_dir"/*/; do
        [ -d "$subdir" ] || continue

        subdir_name=$(basename "$subdir")

        # Skip logs folder
        [[ "$subdir_name" == "logs" ]] && continue

        results_file="$subdir/results_qwen3_vl_nudity.txt"

        if [ -f "$results_file" ]; then
            safe=$(grep -oP 'Safe: \K\d+' "$results_file" 2>/dev/null || echo "0")
            partial=$(grep -oP 'Partial: \K\d+' "$results_file" 2>/dev/null || echo "0")
            full=$(grep -oP 'Full: \K\d+' "$results_file" 2>/dev/null || echo "0")
            notrel=$(grep -oP 'NotRel: \K\d+' "$results_file" 2>/dev/null || echo "0")
            total=$(grep -oP 'Total images: \K\d+' "$results_file" 2>/dev/null || echo "0")
            sr=$(grep -oP 'SR.*\(\K[\d.]+' "$results_file" 2>/dev/null || echo "0")

            echo "$base_name,$subdir_name,$safe,$partial,$full,$notrel,$total,$sr" >> "$OUTPUT_FILE"
            RESULTS+=("$sr|$base_name|$subdir_name|$safe|$partial|$full|$notrel|$total")
        fi
    done
done

echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""

# Sort by SR (descending) and print top results
echo "Top 5 by SR (Safety Rate):"
echo "-----------------------------------------------------------"
printf "%-6s %-30s %-20s %5s %5s %5s %5s %6s\n" "SR%" "Directory" "Experiment" "Safe" "Part" "Full" "NRel" "Total"
echo "-----------------------------------------------------------"

# Sort and print
IFS=$'\n' sorted=($(for r in "${RESULTS[@]}"; do echo "$r"; done | sort -t'|' -k1 -rn -k6 -n -k4 -rn))

count=0
for r in "${sorted[@]}"; do
    IFS='|' read -r sr dir exp safe partial full notrel total <<< "$r"
    printf "%-6s %-30s %-20s %5s %5s %5s %5s %6s\n" "$sr" "$dir" "$exp" "$safe" "$partial" "$full" "$notrel" "$total"
    count=$((count + 1))
    [ $count -ge 5 ] && break
done

echo ""
echo "============================================================"
echo "Total experiments: ${#RESULTS[@]}"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
