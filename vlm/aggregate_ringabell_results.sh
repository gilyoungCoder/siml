#!/bin/bash
# ============================================================================
# Ring-A-Bell Results Aggregation (grouped by method)
#
# Usage:
#   bash aggregate_ringabell_results.sh [--all | --safree | --sdbaseline | ...]
#   (default: --all runs all groups independently)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Directory groups ----

# 1. SAFREE (baseline)
SAFREE_DIRS=(
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/baselines_ringabell/safree"
)

# 2. SD Baseline
SDBASELINE_DIRS=(
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/baselines_ringabell/sd_baseline"
)

# 3. SAFREE + Dual (ours)
SAFREE_DUAL_DIRS=(
    "/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_dual_ringabell_20260129_065829"
)

# 4. SAFREE + Monitor (ours)
SAFREE_MON_DIRS=(
    "/mnt/home/yhgil99/unlearning/SAFREE/results/grid_search_safree_mon_ringabell_20260129_064352"
)

# 5. Dual (ours, standalone)
DUAL_DIRS=(
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260128_201546"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260128_223640"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260129_022351"
)

# 6. Monitor 4class (ours, standalone)
MONITOR_DIRS=(
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_ringabell_20260128_201546"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_mon4class_ringabell_20260129_155025"
)

# 7. Monitor 3class (ours, standalone)
MONITOR_3CLASS_DIRS=(
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_mon3class_ringabell_20260129_160011"
)

# ---- Parse arguments ----
RUN_GROUPS=()
if [ $# -eq 0 ]; then
    RUN_GROUPS=("safree" "sdbaseline" "safree_dual" "safree_mon" "dual" "monitor" "monitor_3class")
else
    for arg in "$@"; do
        case "$arg" in
            --all)              RUN_GROUPS=("safree" "sdbaseline" "safree_dual" "safree_mon" "dual" "monitor" "monitor_3class") ;;
            --safree)           RUN_GROUPS+=("safree") ;;
            --sdbaseline)       RUN_GROUPS+=("sdbaseline") ;;
            --safree-dual)      RUN_GROUPS+=("safree_dual") ;;
            --safree-mon)       RUN_GROUPS+=("safree_mon") ;;
            --dual)             RUN_GROUPS+=("dual") ;;
            --monitor)          RUN_GROUPS+=("monitor") ;;
            --monitor-3class)   RUN_GROUPS+=("monitor_3class") ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

cd "${SCRIPT_DIR}"

# ---- Helper: run aggregation for a group and show best ----
run_group() {
    local group_name="$1"
    shift
    local dirs=("$@")

    echo ""
    echo "============================================================"
    echo "  GROUP: ${group_name}"
    echo "============================================================"

    bash aggregate_qwen3_results.sh "${dirs[@]}"
}

# ---- Run each group ----
for group in "${RUN_GROUPS[@]}"; do
    case "$group" in
        safree)      run_group "SAFREE"          "${SAFREE_DIRS[@]}" ;;
        sdbaseline)  run_group "SD_BASELINE"     "${SDBASELINE_DIRS[@]}" ;;
        safree_dual) run_group "SAFREE+DUAL"     "${SAFREE_DUAL_DIRS[@]}" ;;
        safree_mon)  run_group "SAFREE+MONITOR"  "${SAFREE_MON_DIRS[@]}" ;;
        dual)        run_group "DUAL"            "${DUAL_DIRS[@]}" ;;
        monitor)         run_group "MONITOR_4CLASS"   "${MONITOR_DIRS[@]}" ;;
        monitor_3class)  run_group "MONITOR_3CLASS"   "${MONITOR_3CLASS_DIRS[@]}" ;;
    esac
done

echo ""
echo "============================================================"
echo "All groups completed."
echo "============================================================"
