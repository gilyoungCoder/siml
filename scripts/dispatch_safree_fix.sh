#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning

echo "[$(date)] SAFREE fix dispatch"

# GPU 0: nudity_rab + nudity_ud
(
  bash $REPO/scripts/run_safree_official.sh 0 nudity_rab 2>&1 | tee -a $REPO/logs/safree_nudity_rab.log
  bash $REPO/scripts/run_safree_official.sh 0 nudity_ud  2>&1 | tee -a $REPO/logs/safree_nudity_ud.log
) &

# GPU 1: nudity_p4dn + nudity_mma
(
  bash $REPO/scripts/run_safree_official.sh 1 nudity_p4dn 2>&1 | tee -a $REPO/logs/safree_nudity_p4dn.log
  bash $REPO/scripts/run_safree_official.sh 1 nudity_mma  2>&1 | tee -a $REPO/logs/safree_nudity_mma.log
) &

# GPU 2: nudity_i2p
(
  bash $REPO/scripts/run_safree_official.sh 2 nudity_i2p 2>&1 | tee -a $REPO/logs/safree_nudity_i2p.log
) &

wait
echo "[$(date)] SAFREE fix all done"
