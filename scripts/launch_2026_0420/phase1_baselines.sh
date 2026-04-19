#!/usr/bin/env bash
# phase1_baselines.sh — Launch 15 baseline jobs (3 backbone × 5 dataset)
#
# Allowed GPUs (HARD POLICY — DO NOT MODIFY):
#   siml-01 SIML01_GPUS=(0 1 2 3 4 5 6 7)
#   siml-06 SIML06_GPUS=(4 5 6 7)
#   siml-08 SIML08_GPUS=(4 5)
#   siml-09 SIML09_GPUS=(0)
#
# Each job uses CUDA_VISIBLE_DEVICES=<single_id> and --device cuda:0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gpu_policy.sh"

LOG_DIR="${REPO}/logs/launch_0420"
mkdir -p "${LOG_DIR}"

# ── Dataset definitions ──
declare -A PROMPT_FILES
PROMPT_FILES[mja_sexual]="${REPO}/CAS_SpatialCFG/prompts/mja_sexual.txt"
PROMPT_FILES[mja_violent]="${REPO}/CAS_SpatialCFG/prompts/mja_violent.txt"
PROMPT_FILES[mja_disturbing]="${REPO}/CAS_SpatialCFG/prompts/mja_disturbing.txt"
PROMPT_FILES[mja_illegal]="${REPO}/CAS_SpatialCFG/prompts/mja_illegal.txt"
PROMPT_FILES[rab]="${REPO}/CAS_SpatialCFG/prompts/nudity-ring-a-bell.csv"

DATASETS=(mja_sexual mja_violent mja_disturbing mja_illegal rab)

# ── Output dirs ──
SD14_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_sd14"
SD3_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_sd3"
FLUX1_OUTBASE="${REPO}/CAS_SpatialCFG/outputs/launch_0420/baseline_flux1"

# ── Generator scripts ──
SD14_GEN="${REPO}/CAS_SpatialCFG/generate_baseline.py"
SD3_GEN="${REPO}/scripts/sd3/generate_sd3_baseline.py"
FLUX1_GEN="${REPO}/CAS_SpatialCFG/generate_flux1_v1.py"

echo "============================================================"
echo "Phase 1 Baselines — Pre-launch GPU checks"
echo "============================================================"

# ── Pre-launch GPU verification ──
# siml-01: GPUs 0-4 (5 datasets)
echo "--- siml-01 (SD1.4 GPUs 0-4) ---"
for gpu in 0 1 2 3 4; do
    check_gpu_free siml-01 "$gpu" || { echo "ABORT: siml-01 GPU $gpu busy. Stopping."; exit 1; }
done

# siml-06: GPUs 4,5,6,7
echo "--- siml-06 (SD3 GPUs 4,5,6,7) ---"
for gpu in 4 5 6 7; do
    check_gpu_free siml-06 "$gpu" || { echo "ABORT: siml-06 GPU $gpu busy. Stopping."; exit 1; }
done

# siml-09: GPU 0 (FLUX1 H100)
echo "--- siml-09 (FLUX1 GPU 0) ---"
check_gpu_free siml-09 0 || { echo "ABORT: siml-09 GPU 0 busy. Stopping."; exit 1; }

# siml-08: GPUs 4,5 (FLUX1 A6000)
echo "--- siml-08 (FLUX1 GPUs 4,5) ---"
for gpu in 4 5; do
    check_gpu_free siml-08 "$gpu" || { echo "ABORT: siml-08 GPU $gpu busy. Stopping."; exit 1; }
done

echo ""
echo "All GPU checks passed. Launching jobs..."
echo ""

# ── SD1.4 — siml-01 GPUs 0-4, round-robin across 5 datasets ──
echo "============================================================"
echo "Launching SD1.4 baselines on siml-01"
echo "============================================================"

SD14_GPUS=(0 1 2 3 4)
for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    gpu="${SD14_GPUS[$i]}"
    prompts="${PROMPT_FILES[$ds]}"
    outdir="${SD14_OUTBASE}/${ds}"
    logfile="${LOG_DIR}/baseline_sd14_${ds}_siml-01g${gpu}.log"

    echo "  [SD1.4] ${ds} -> siml-01 GPU ${gpu} | log: ${logfile}"
    ssh siml-01 "cd ${REPO} && CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} ${SD14_GEN} \
        --prompts ${prompts} \
        --outdir ${outdir} \
        --steps 50 \
        > ${logfile} 2>&1 &
    echo \$!"
done

echo ""

# ── SD3 — siml-06 GPUs 4,5,6,7 ──
# mja_sexual->GPU4, mja_violent->GPU5, mja_disturbing->GPU6, mja_illegal->GPU7
# rab->GPU4 sequential after mja_sexual (piped as second command)
echo "============================================================"
echo "Launching SD3 baselines on siml-06"
echo "============================================================"

SD3_DATASET_GPU=(mja_sexual:4 mja_violent:5 mja_disturbing:6 mja_illegal:7)
for entry in "${SD3_DATASET_GPU[@]}"; do
    ds="${entry%%:*}"
    gpu="${entry##*:}"
    prompts="${PROMPT_FILES[$ds]}"
    outdir="${SD3_OUTBASE}/${ds}"
    logfile="${LOG_DIR}/baseline_sd3_${ds}_siml-06g${gpu}.log"

    echo "  [SD3] ${ds} -> siml-06 GPU ${gpu} | log: ${logfile}"
    ssh siml-06 "cd ${REPO} && CUDA_VISIBLE_DEVICES=${gpu} nohup ${PYTHON} ${SD3_GEN} \
        --prompts ${prompts} \
        --outdir ${outdir} \
        --device cuda:0 \
        --no_cpu_offload \
        > ${logfile} 2>&1 &
    echo \$!"
done

# rab on GPU4 — sequential after mja_sexual; launch as second nohup job
# (will wait for mja_sexual since same GPU; dispatched separately as background)
RAB_DS=rab
RAB_GPU=4
RAB_PROMPTS="${PROMPT_FILES[$RAB_DS]}"
RAB_OUTDIR="${SD3_OUTBASE}/${RAB_DS}"
RAB_LOG="${LOG_DIR}/baseline_sd3_rab_siml-06g${RAB_GPU}.log"
echo "  [SD3] rab -> siml-06 GPU ${RAB_GPU} (queued sequential) | log: ${RAB_LOG}"
ssh siml-06 "cd ${REPO} && nohup bash -c '\
    while pgrep -f \"CUDA_VISIBLE_DEVICES=${RAB_GPU}.*generate_sd3_baseline\" > /dev/null; do sleep 30; done; \
    CUDA_VISIBLE_DEVICES=${RAB_GPU} ${PYTHON} ${SD3_GEN} \
        --prompts ${RAB_PROMPTS} \
        --outdir ${RAB_OUTDIR} \
        --device cuda:0 \
        --no_cpu_offload \
    ' > ${RAB_LOG} 2>&1 &
echo \$!"

echo ""

# ── FLUX1 — siml-09 GPU 0 (H100, cpu_offload on) + siml-08 GPU 4,5 ──
# 5 datasets across 3 GPU slots. siml-09 handles 2 (sequential), siml-08 GPU4 and GPU5 get 1 each.
# Distribution:
#   siml-09 GPU0: mja_sexual (first), mja_violent (sequential after)
#   siml-08 GPU4: mja_disturbing
#   siml-08 GPU5: mja_illegal
#   siml-09 GPU0 (3rd, after mja_violent): rab
# Note: generate_flux1_v1.py always uses enable_model_cpu_offload(gpu_id=gpu_id)
#       gpu_id is parsed from --device (cuda:0 -> gpu_id=0). CUDA_VISIBLE_DEVICES remaps
#       the physical GPU to index 0, so --device cuda:0 always correct.
echo "============================================================"
echo "Launching FLUX1 baselines on siml-09 (H100) + siml-08 (A6000)"
echo "============================================================"

# siml-09 GPU0: mja_sexual
DS=mja_sexual
GPU=0
LOG="${LOG_DIR}/baseline_flux1_${DS}_siml-09g${GPU}.log"
OUTDIR="${FLUX1_OUTBASE}/${DS}"
echo "  [FLUX1] ${DS} -> siml-09 GPU ${GPU} | log: ${LOG}"
ssh siml-09 "cd ${REPO} && CUDA_VISIBLE_DEVICES=${GPU} nohup ${PYTHON} ${FLUX1_GEN} \
    --prompts ${PROMPT_FILES[$DS]} \
    --outdir ${OUTDIR} \
    --no_safety \
    --height 1024 --width 1024 \
    --device cuda:0 \
    > ${LOG} 2>&1 &
echo \$!"

# siml-09 GPU0: mja_violent (sequential after mja_sexual)
DS2=mja_violent
LOG2="${LOG_DIR}/baseline_flux1_${DS2}_siml-09g${GPU}.log"
OUTDIR2="${FLUX1_OUTBASE}/${DS2}"
echo "  [FLUX1] ${DS2} -> siml-09 GPU ${GPU} (sequential) | log: ${LOG2}"
ssh siml-09 "cd ${REPO} && nohup bash -c '\
    while pgrep -f \"CUDA_VISIBLE_DEVICES=${GPU}.*generate_flux1_v1\" > /dev/null; do sleep 30; done; \
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${FLUX1_GEN} \
        --prompts ${PROMPT_FILES[$DS2]} \
        --outdir ${OUTDIR2} \
        --no_safety \
        --height 1024 --width 1024 \
        --device cuda:0 \
    ' > ${LOG2} 2>&1 &
echo \$!"

# siml-09 GPU0: rab (sequential after mja_violent, 3rd slot)
DS3=rab
LOG3="${LOG_DIR}/baseline_flux1_${DS3}_siml-09g${GPU}.log"
OUTDIR3="${FLUX1_OUTBASE}/${DS3}"
echo "  [FLUX1] ${DS3} -> siml-09 GPU ${GPU} (sequential, 3rd) | log: ${LOG3}"
ssh siml-09 "cd ${REPO} && nohup bash -c '\
    # Wait for both mja_sexual and mja_violent to finish
    while pgrep -f \"CUDA_VISIBLE_DEVICES=${GPU}.*generate_flux1_v1\" > /dev/null; do sleep 30; done; \
    sleep 5; \
    while pgrep -f \"CUDA_VISIBLE_DEVICES=${GPU}.*generate_flux1_v1\" > /dev/null; do sleep 30; done; \
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${FLUX1_GEN} \
        --prompts ${PROMPT_FILES[$DS3]} \
        --outdir ${OUTDIR3} \
        --no_safety \
        --height 1024 --width 1024 \
        --device cuda:0 \
    ' > ${LOG3} 2>&1 &
echo \$!"

# siml-08 GPU4: mja_disturbing
DS=mja_disturbing
GPU=4
LOG="${LOG_DIR}/baseline_flux1_${DS}_siml-08g${GPU}.log"
OUTDIR="${FLUX1_OUTBASE}/${DS}"
echo "  [FLUX1] ${DS} -> siml-08 GPU ${GPU} | log: ${LOG}"
ssh siml-08 "cd ${REPO} && CUDA_VISIBLE_DEVICES=${GPU} nohup ${PYTHON} ${FLUX1_GEN} \
    --prompts ${PROMPT_FILES[$DS]} \
    --outdir ${OUTDIR} \
    --no_safety \
    --height 1024 --width 1024 \
    --device cuda:0 \
    > ${LOG} 2>&1 &
echo \$!"

# siml-08 GPU5: mja_illegal
DS=mja_illegal
GPU=5
LOG="${LOG_DIR}/baseline_flux1_${DS}_siml-08g${GPU}.log"
OUTDIR="${FLUX1_OUTBASE}/${DS}"
echo "  [FLUX1] ${DS} -> siml-08 GPU ${GPU} | log: ${LOG}"
ssh siml-08 "cd ${REPO} && CUDA_VISIBLE_DEVICES=${GPU} nohup ${PYTHON} ${FLUX1_GEN} \
    --prompts ${PROMPT_FILES[$DS]} \
    --outdir ${OUTDIR} \
    --no_safety \
    --height 1024 --width 1024 \
    --device cuda:0 \
    > ${LOG} 2>&1 &
echo \$!"

echo ""
echo "============================================================"
echo "All 15 baseline jobs dispatched."
echo "Run monitor_phase1.sh to check status."
echo "============================================================"
