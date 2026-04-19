#!/usr/bin/env bash
# monitor_phase1.sh — On-demand status check for Phase 1 baseline jobs.
# Run any time to see running processes and completed image counts.

REPO=/mnt/home3/yhgil99/unlearning
SD14_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_sd14"
SD3_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_sd3"
FLUX1_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_flux1"
DATASETS=(mja_sexual mja_violent mja_disturbing mja_illegal rab)

echo "========================================================"
echo "Phase 1 Monitor — $(date)"
echo "========================================================"

for host in siml-01 siml-06 siml-08 siml-09; do
    echo ""
    echo "--- ${host}: running generate_ processes ---"
    ssh "$host" "pgrep -af 'generate_' 2>/dev/null || echo '  (none)'"
done

echo ""
echo "========================================================"
echo "Completed image counts (PNG files per output dir)"
echo "========================================================"

printf "%-12s %-14s %-6s\n" "BACKBONE" "DATASET" "IMGS"
printf "%-12s %-14s %-6s\n" "--------" "-------" "----"

for ds in "${DATASETS[@]}"; do
    for backbone in sd14 sd3 flux1; do
        case "$backbone" in
            sd14)  host=siml-01; outbase="$SD14_OUTBASE" ;;
            sd3)   host=siml-06; outbase="$SD3_OUTBASE" ;;
            flux1) host=siml-09; outbase="$FLUX1_OUTBASE" ;;
        esac
        count=$(ssh "$host" "ls ${outbase}/${ds}/*.png 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        count=$(echo "$count" | tr -d ' ')
        printf "%-12s %-14s %-6s\n" "$backbone" "$ds" "${count:-0}"
    done
done

echo ""
echo "========================================================"
echo "Log tail (last 5 lines per log)"
echo "========================================================"
LOG_DIR="${REPO}/logs/launch_0420"
for host in siml-01 siml-06 siml-08 siml-09; do
    echo ""
    echo "--- ${host} logs ---"
    ssh "$host" "for f in ${LOG_DIR}/baseline_*_${host/siml-0/siml-0}*.log; do
        [ -f \"\$f\" ] || continue
        echo \"  >> \$(basename \$f)\";
        tail -3 \"\$f\" 2>/dev/null | sed 's/^/     /';
    done" 2>/dev/null || echo "  (no logs yet)"
done
