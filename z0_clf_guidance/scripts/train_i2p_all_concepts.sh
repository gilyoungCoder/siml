#!/bin/bash
# ============================================================================
# [Z0] Launch ALL 8 training jobs in parallel on GPU 0-7
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
mkdir -p logs

echo "Launching all z0 classifier training jobs..."
echo ""

bash scripts/train.sh
bash scripts/train_i2p_violence.sh
bash scripts/train_i2p_harassment.sh
bash scripts/train_i2p_hate.sh
bash scripts/train_i2p_illegal.sh
bash scripts/train_i2p_selfharm.sh
bash scripts/train_i2p_shocking.sh
bash scripts/train_i2p_sexual.sh

echo ""
echo "=============================================="
echo "All 8 jobs launched! Monitor with:"
echo "  tail -f logs/train_*.log"
echo "=============================================="
