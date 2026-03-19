#!/bin/bash
# Aggregate mon01_grid FID results (coco configs)
# Usage: bash vlm/aggregate_mon01_grid_fid.sh

set -u

BASE="SoftDelete+CG/scg_outputs/mon01_grid/coco"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${SCRIPT_DIR}/aggregated_mon01_grid_fid_${TIMESTAMP}.csv"

echo "============================================================"
echo "mon01_grid FID Aggregation"
echo "============================================================"
echo ""

echo "config,fid,n_images" > "$OUTPUT_FILE"

declare -a RANKED=()

for config_dir in "$BASE"/*/; do
    [ -d "$config_dir" ] || continue
    config=$(basename "$config_dir")
    [[ "$config" == "logs" ]] && continue

    metrics_file="$config_dir/eval_metrics.json"
    [ -f "$metrics_file" ] || continue

    fid=$(python3 -c "import json; d=json.load(open('$metrics_file')); print(d.get('fid', 'N/A'))" 2>/dev/null)
    n_images=$(python3 -c "import json; d=json.load(open('$metrics_file')); print(d.get('n_images', 'N/A'))" 2>/dev/null)

    [ -z "$fid" ] && continue

    echo "$config,$fid,$n_images" >> "$OUTPUT_FILE"
    RANKED+=("${fid}|${config}")
done

# Sort by FID ascending (lower is better)
IFS=$'\n' sorted=($(for r in "${RANKED[@]}"; do echo "$r"; done | sort -t'|' -k1 -n))

echo "Rank | Config                                    | FID      | "
echo "-----|-------------------------------------------|----------|"

count=0
for r in "${sorted[@]}"; do
    IFS='|' read -r fid config <<< "$r"
    count=$((count + 1))
    printf " %2d  | %-41s | %8.2f |\n" "$count" "$config" "$fid"
done

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
