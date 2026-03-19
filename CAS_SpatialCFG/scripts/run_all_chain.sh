#!/bin/bash
# ============================================================
# Master chain: Wait for V4 → Run V5 → Run AMG extended
# All on Ring-A-Bell, all on this server
# ============================================================

echo "=== Master Chain Started $(date) ==="

# Wait for V4 massive to finish
V4_PID=$(pgrep -f "run_v4_massive.sh" 2>/dev/null | head -1)
if [ -n "$V4_PID" ]; then
    echo "Waiting for V4 massive (PID: $V4_PID) to finish..."
    while kill -0 "$V4_PID" 2>/dev/null; do
        sleep 60
    done
    echo "V4 massive finished! $(date)"
else
    echo "V4 massive not running (already done or not started)"
fi

# Run V5
echo ""
echo "============================================================"
echo ">>> Starting V5 Massive Grid Search $(date)"
echo "============================================================"
bash /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/scripts/run_v5_massive.sh

# Run AMG extended
echo ""
echo "============================================================"
echo ">>> Starting AMG Extended $(date)"
echo "============================================================"
bash /mnt/home/yhgil99/unlearning/AMG/scripts/run_amg_extended.sh

echo ""
echo "============================================================"
echo "ALL CHAINS COMPLETE! $(date)"
echo "============================================================"
