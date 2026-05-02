#!/bin/bash
# ============================================================================
# WAVE 2: I2P concepts + COCO + safree/mma
# Run after WAVE 1 finishes (check with: ps aux | grep generate_sd3)
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SCRIPT_DIR="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUTPUT_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG_DIR="/mnt/home3/yhgil99/unlearning/logs/sd3"
PROMPT_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
I2P_DIR="${PROMPT_DIR}/i2p_categories"

STEPS=28; CFG=7.0; RES=1024; SEED=42

launch() {
    local GPU=$1 METHOD=$2 DATASET=$3 PROMPT=$4 CONCEPT=$5
    local OUTDIR="${OUTPUT_BASE}/${METHOD}/${DATASET}"
    local LOG="${LOG_DIR}/${METHOD}_${DATASET}.log"
    echo "[GPU${GPU}] ${METHOD}/${DATASET}"
    if [ "$METHOD" = "baseline" ]; then
        CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON "${SCRIPT_DIR}/generate_sd3_baseline.py" \
            --prompts "$PROMPT" --outdir "$OUTDIR" \
            --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
            > "$LOG" 2>&1 &
    elif [ "$METHOD" = "safree" ]; then
        CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON "${SCRIPT_DIR}/generate_sd3_safree.py" \
            --prompts "$PROMPT" --outdir "$OUTDIR" --concept "$CONCEPT" \
            --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
            > "$LOG" 2>&1 &
    fi
    echo "  PID: $!"
}

echo "=============================================="
echo "SD3 WAVE 2: I2P + COCO + safree/mma"
echo "=============================================="

# ── Batch A: Baselines for I2P (7 concepts) ──
launch 0 baseline i2p_sexual     "${I2P_DIR}/i2p_sexual.csv" ""
launch 1 baseline i2p_violence   "${I2P_DIR}/i2p_violence.csv" ""
launch 2 baseline i2p_harassment "${I2P_DIR}/i2p_harassment.csv" ""
launch 3 baseline i2p_hate       "${I2P_DIR}/i2p_hate.csv" ""
launch 5 baseline i2p_shocking   "${I2P_DIR}/i2p_shocking.csv" ""
launch 6 baseline i2p_illegal    "${I2P_DIR}/i2p_illegal_activity.csv" ""
launch 7 baseline i2p_selfharm   "${I2P_DIR}/i2p_self-harm.csv" ""

echo ""
echo "Batch A launched (7 I2P baselines). Wait for completion, then run Batch B:"
echo ""
echo "# Batch B: SAFREE for I2P + COCO + safree/mma"
echo "# launch 0 safree i2p_sexual     \"\${I2P_DIR}/i2p_sexual.csv\" sexual"
echo "# launch 1 safree i2p_violence   \"\${I2P_DIR}/i2p_violence.csv\" violence"
echo "# launch 2 safree i2p_harassment \"\${I2P_DIR}/i2p_harassment.csv\" harassment"
echo "# launch 3 safree i2p_hate       \"\${I2P_DIR}/i2p_hate.csv\" hate"
echo "# launch 5 safree i2p_shocking   \"\${I2P_DIR}/i2p_shocking.csv\" shocking"
echo "# launch 6 safree i2p_illegal    \"\${I2P_DIR}/i2p_illegal_activity.csv\" illegal"
echo "# launch 7 safree i2p_selfharm   \"\${I2P_DIR}/i2p_self-harm.csv\" selfharm"
echo ""
echo "# Batch C: COCO (quality eval) + safree/mma"
echo "# launch 0 baseline coco \"\${PROMPT_DIR}/coco_30k_10k.csv\" \"\""
echo "# launch 1 safree   coco \"\${PROMPT_DIR}/coco_30k_10k.csv\" none"
echo "# launch 2 safree   mma  \"\${PROMPT_DIR}/mma-diffusion-nsfw-adv-prompts.csv\" sexual"
