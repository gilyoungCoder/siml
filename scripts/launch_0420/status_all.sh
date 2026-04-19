#!/usr/bin/env bash
# status_all.sh — Monitor all 15 workers. Run from any host with SSH access.
#
# Usage: bash scripts/launch_0420/status_all.sh

REPO=/mnt/home3/yhgil99/unlearning
LOG_ROOT=${REPO}/logs/launch_0420
MANIFEST_DIR=${REPO}/scripts/launch_0420/manifests
OUT_ROOT=${REPO}/CAS_SpatialCFG/outputs/launch_0420

declare -a WORKERS=(
    "siml-01 0" "siml-01 1" "siml-01 2" "siml-01 3"
    "siml-01 4" "siml-01 5" "siml-01 6" "siml-01 7"
    "siml-06 4" "siml-06 5" "siml-06 6" "siml-06 7"
    "siml-08 4" "siml-08 5"
    "siml-09 0"
)

printf "%-12s %-5s %-8s %-8s %-10s %-10s %s\n" \
    "HOST" "GPU" "TOTAL" "DONE" "SKIPPED" "FAILED" "STATUS"
printf '%s\n' "$(printf '─%.0s' {1..75})"

for worker in "${WORKERS[@]}"; do
    HOST=$(echo $worker | awk '{print $1}')
    GPU=$(echo $worker | awk '{print $2}')
    LOG="${LOG_ROOT}/worker_${HOST}_g${GPU}.log"
    MANIFEST="${MANIFEST_DIR}/worker_${HOST}_g${GPU}.csv"

    INFO=$(ssh -o ConnectTimeout=8 "${HOST}" "
        # Count total jobs (subtract header)
        total=0
        done_count=0
        skipped_count=0
        failed_count=0
        status='running'

        if [[ -f '${MANIFEST}' ]]; then
            total=\$((\$(wc -l < '${MANIFEST}') - 1))
        fi

        if [[ -f '${LOG}' ]]; then
            done_count=\$(grep -c '\[GEN OK\]\|\[SKIP gen\]' '${LOG}' 2>/dev/null || echo 0)
            skipped_count=\$(grep -c '\[SKIP gen\]' '${LOG}' 2>/dev/null || echo 0)
            failed_count=\$(grep -c '\[GEN FAILED\]' '${LOG}' 2>/dev/null || echo 0)
            if grep -q 'Worker.*FINISHED' '${LOG}' 2>/dev/null; then
                status='DONE'
            fi
        fi

        # Check if worker process is still running
        if pgrep -f 'worker.sh ${HOST} ${GPU}' > /dev/null 2>&1; then
            status='running'
        elif [[ \"\$status\" != 'DONE' ]]; then
            status='stopped?'
        fi

        echo \"\${total} \${done_count} \${skipped_count} \${failed_count} \${status}\"
    " 2>/dev/null || echo "0 0 0 0 ssh_err")

    TOTAL=$(echo $INFO | awk '{print $1}')
    DONE=$(echo $INFO | awk '{print $2}')
    SKIP=$(echo $INFO | awk '{print $3}')
    FAIL=$(echo $INFO | awk '{print $4}')
    STATUS=$(echo $INFO | awk '{print $5}')

    printf "%-12s %-5s %-8s %-8s %-10s %-10s %s\n" \
        "${HOST}" "${GPU}" "${TOTAL}" "${DONE}" "${SKIP}" "${FAIL}" "${STATUS}"
done

echo ""
echo "Logs: ${LOG_ROOT}/"
echo "Run again to refresh. Timestamp: $(date)"
