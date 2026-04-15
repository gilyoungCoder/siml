#!/bin/bash
# ============================================================================
# Launch ALL SD3 experiments on siml-01 (GPUs 1,2,3,5,6,7)
# GPU0: already running baseline/rab
# GPU4: occupied by other user — DO NOT USE
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SCRIPT_DIR="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUTPUT_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG_DIR="/mnt/home3/yhgil99/unlearning/logs/sd3"
PROMPT_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"

mkdir -p "$LOG_DIR"

STEPS=28
CFG=7.0
RES=1024
SEED=42

launch() {
    local GPU=$1
    local METHOD=$2
    local DATASET=$3
    local PROMPT=$4
    local CONCEPT=$5
    local EXTRA_ARGS=$6

    local OUTDIR="${OUTPUT_BASE}/${METHOD}/${DATASET}"
    local LOG="${LOG_DIR}/${METHOD}_${DATASET}.log"

    echo "[GPU${GPU}] ${METHOD}/${DATASET} -> ${LOG}"

    if [ "$METHOD" = "baseline" ]; then
        CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON "${SCRIPT_DIR}/generate_sd3_baseline.py" \
            --prompts "$PROMPT" --outdir "$OUTDIR" \
            --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
            $EXTRA_ARGS > "$LOG" 2>&1 &
    elif [ "$METHOD" = "safree" ]; then
        CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON "${SCRIPT_DIR}/generate_sd3_safree.py" \
            --prompts "$PROMPT" --outdir "$OUTDIR" \
            --concept "$CONCEPT" \
            --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
            $EXTRA_ARGS > "$LOG" 2>&1 &
    fi

    echo "  PID: $!"
}

echo "=============================================="
echo "SD3 Mass Launch on SIML-01"
echo "=============================================="
echo ""

# ── WAVE 1: Baselines (nudity datasets) ──
# GPU0: baseline/rab — ALREADY RUNNING
launch 1 baseline unlearndiff "${PROMPT_DIR}/unlearn_diff_nudity.csv" "" ""
launch 2 baseline p4dn        "${PROMPT_DIR}/p4dn_16_prompt.csv" "" ""
launch 3 baseline mma         "${PROMPT_DIR}/mma-diffusion-nsfw-adv-prompts.csv" "" ""

# ── WAVE 1: SAFREE (nudity datasets) ──
launch 5 safree rab        "${PROMPT_DIR}/nudity-ring-a-bell.csv" "sexual" ""
launch 6 safree unlearndiff "${PROMPT_DIR}/unlearn_diff_nudity.csv" "sexual" ""
launch 7 safree p4dn        "${PROMPT_DIR}/p4dn_16_prompt.csv" "sexual" ""

echo ""
echo "=============================================="
echo "WAVE 1 launched! (6 jobs on GPUs 1,2,3,5,6,7)"
echo "GPU0: baseline/rab (already running)"
echo "GPU4: RESERVED (other user)"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo "Check:   nvidia-smi"
echo "=============================================="
echo ""
echo "WAVE 2 (queued after WAVE 1 finishes):"
echo "  - safree/mma (1000 prompts, longest)"
echo "  - baseline + safree for I2P concepts (7 categories)"
echo "  - baseline + safree for COCO (quality check)"
echo ""
echo "Run WAVE 2 manually after WAVE 1 completes:"
echo "  bash ${SCRIPT_DIR}/launch_wave2_siml01.sh"
