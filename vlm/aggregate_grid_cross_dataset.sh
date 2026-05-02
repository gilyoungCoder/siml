#!/bin/bash
# ============================================================================
# Aggregate grid search results across datasets by config
# Finds the same config (experiment name) across multiple dataset folders
# and computes average SR.
#
# Usage:
#   bash aggregate_grid_cross_dataset.sh <dir1> <dir2> <dir3> ...
# ============================================================================

set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <grid_result_dir1> [dir2] ..."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${SCRIPT_DIR}/aggregated_grid_cross_dataset_${TIMESTAMP}.csv"

echo "============================================================"
echo "Cross-Dataset Grid Search Aggregation"
echo "============================================================"
echo ""

# Collect: config -> dataset -> values
declare -A CONFIG_SR
declare -A CONFIG_SAFE
declare -A CONFIG_PARTIAL
declare -A CONFIG_FULL
declare -A CONFIG_NOTREL
declare -A CONFIG_TOTAL
declare -A CONFIG_DATASETS
declare -A DATASET_NAMES

for base_dir in "$@"; do
    [ -d "$base_dir" ] || continue
    ds_name=$(basename "$base_dir")
    DATASET_NAMES[$ds_name]=1
    echo "Scanning: $ds_name"

    for subdir in "$base_dir"/*/; do
        [ -d "$subdir" ] || continue
        config=$(basename "$subdir")
        [[ "$config" == "logs" ]] && continue

        results_file="$subdir/results_qwen3_vl_nudity.txt"
        [ -f "$results_file" ] || continue

        sr=$(grep -oP 'SR.*\(\K[\d.]+' "$results_file" 2>/dev/null || echo "")
        [ -z "$sr" ] && continue

        safe=$(grep -oP 'Safe: \K\d+' "$results_file" 2>/dev/null || echo "0")
        partial=$(grep -oP 'Partial: \K\d+' "$results_file" 2>/dev/null || echo "0")
        full=$(grep -oP 'Full: \K\d+' "$results_file" 2>/dev/null || echo "0")
        notrel=$(grep -oP 'NotRel: \K\d+' "$results_file" 2>/dev/null || echo "0")
        total=$(grep -oP 'Total images: \K\d+' "$results_file" 2>/dev/null || echo "0")

        CONFIG_SR["${config}|${ds_name}"]="$sr"
        CONFIG_SAFE["${config}|${ds_name}"]="$safe"
        CONFIG_PARTIAL["${config}|${ds_name}"]="$partial"
        CONFIG_FULL["${config}|${ds_name}"]="$full"
        CONFIG_NOTREL["${config}|${ds_name}"]="$notrel"
        CONFIG_TOTAL["${config}|${ds_name}"]="$total"

        if [ -z "${CONFIG_DATASETS[$config]:-}" ]; then
            CONFIG_DATASETS[$config]="$ds_name"
        else
            CONFIG_DATASETS[$config]="${CONFIG_DATASETS[$config]} $ds_name"
        fi
    done
done

DS_LIST=($(for k in "${!DATASET_NAMES[@]}"; do echo "$k"; done | sort))
NUM_DS=${#DS_LIST[@]}

echo ""
echo "Datasets found: ${DS_LIST[*]}"
echo "Total configs: ${#CONFIG_DATASETS[@]}"
echo ""

# CSV header
csv_h="config"
for ds in "${DS_LIST[@]}"; do
    csv_h="${csv_h},${ds}_sr,${ds}_safe,${ds}_part,${ds}_full,${ds}_nrel,${ds}_total"
done
csv_h="${csv_h},avg_sr,num_datasets"
echo "$csv_h" > "$OUTPUT_FILE"

# Compute avg SR per config
declare -a RANKED=()

for config in "${!CONFIG_DATASETS[@]}"; do
    sr_sum=0
    count=0
    row="$config"

    for ds in "${DS_LIST[@]}"; do
        sr="${CONFIG_SR["${config}|${ds}"]:-}"
        safe="${CONFIG_SAFE["${config}|${ds}"]:-}"
        partial="${CONFIG_PARTIAL["${config}|${ds}"]:-}"
        full="${CONFIG_FULL["${config}|${ds}"]:-}"
        notrel="${CONFIG_NOTREL["${config}|${ds}"]:-}"
        total="${CONFIG_TOTAL["${config}|${ds}"]:-}"

        if [ -n "$sr" ]; then
            row="${row},${sr},${safe},${partial},${full},${notrel},${total}"
            sr_sum=$(awk "BEGIN {print $sr_sum + $sr}")
            count=$((count + 1))
        else
            row="${row},-,-,-,-,-,-"
        fi
    done

    if [ "$count" -gt 0 ]; then
        avg=$(awk "BEGIN {printf \"%.1f\", $sr_sum / $count}")
    else
        avg="0.0"
    fi
    row="${row},${avg},${count}"
    echo "$row" >> "$OUTPUT_FILE"
    RANKED+=("${avg}|${count}|${config}")
done

# Sort by avg SR descending
IFS=$'\n' sorted=($(for r in "${RANKED[@]}"; do echo "$r"; done | sort -t'|' -k1 -rn -k2 -rn))

echo "============================================================"
echo "Top 20 configs by Average SR% (across ${NUM_DS} datasets)"
echo "============================================================"
echo ""

# Print per-dataset detail for top 20
count=0
for r in "${sorted[@]}"; do
    IFS='|' read -r avg num_ds config <<< "$r"
    count=$((count + 1))

    printf "[%2d] %-45s  AvgSR: %s\n" "$count" "$config" "$avg"

    for ds in "${DS_LIST[@]}"; do
        sr="${CONFIG_SR["${config}|${ds}"]:-"-"}"
        safe="${CONFIG_SAFE["${config}|${ds}"]:-"-"}"
        partial="${CONFIG_PARTIAL["${config}|${ds}"]:-"-"}"
        full="${CONFIG_FULL["${config}|${ds}"]:-"-"}"
        notrel="${CONFIG_NOTREL["${config}|${ds}"]:-"-"}"
        total="${CONFIG_TOTAL["${config}|${ds}"]:-"-"}"
        printf "     %-12s  SR=%5s  Safe=%4s  Part=%4s  Full=%4s  NRel=%4s  Total=%4s\n" \
            "$ds" "$sr" "$safe" "$partial" "$full" "$notrel" "$total"
    done
    echo ""

    [ $count -ge 20 ] && break
done

echo "============================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
