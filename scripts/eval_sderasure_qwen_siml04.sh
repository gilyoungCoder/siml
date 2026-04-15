#!/bin/bash
# SDErasure Qwen3-VL evaluation on siml-04 GPU 5
# Nudity 4 datasets + Multi-concept 6 datasets

set -e
export CUDA_VISIBLE_DEVICES=5
cd /mnt/home3/yhgil99/unlearning

PYTHON=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL_SCRIPT=vlm/opensource_vlm_i2p_all.py
BASE=unlearning-baselines/outputs

echo "======================================================"
echo " SDErasure Qwen3-VL Evaluation — GPU 5"
echo " Started: $(date)"
echo "======================================================"

# --- Part 1: Nudity datasets ---
echo ""
echo "========== [1/10] SDErasure nudity — ringabell =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure_nudity/ringabell nudity qwen
echo "[1/10] Done: $(date)"

echo ""
echo "========== [2/10] SDErasure nudity — mma =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure_nudity/mma nudity qwen
echo "[2/10] Done: $(date)"

echo ""
echo "========== [3/10] SDErasure nudity — p4dn =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure_nudity/p4dn nudity qwen
echo "[3/10] Done: $(date)"

echo ""
echo "========== [4/10] SDErasure nudity — unlearndiff =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure_nudity/unlearndiff nudity qwen
echo "[4/10] Done: $(date)"

# --- Part 2: Multi-concept datasets ---
echo ""
echo "========== [5/10] SDErasure — violence =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/violence violence qwen
echo "[5/10] Done: $(date)"

echo ""
echo "========== [6/10] SDErasure — harassment =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/harassment harassment qwen
echo "[6/10] Done: $(date)"

echo ""
echo "========== [7/10] SDErasure — hate =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/hate hate qwen
echo "[7/10] Done: $(date)"

echo ""
echo "========== [8/10] SDErasure — shocking =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/shocking shocking qwen
echo "[8/10] Done: $(date)"

echo ""
echo "========== [9/10] SDErasure — illegal_activity =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/illegal_activity illegal qwen
echo "[9/10] Done: $(date)"

echo ""
echo "========== [10/10] SDErasure — self_harm =========="
$PYTHON $EVAL_SCRIPT $BASE/sderasure/self_harm self_harm qwen
echo "[10/10] Done: $(date)"

echo ""
echo "======================================================"
echo " ALL DONE: $(date)"
echo "======================================================"
