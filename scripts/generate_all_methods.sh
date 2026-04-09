#!/bin/bash
# ============================================================================
# Unified Generation Script: All Methods × All Datasets
#
# Methods:
#   1. sd_baseline     - Vanilla Stable Diffusion v1.4
#   2. safree          - SAFREE only
#   3. ours_dual       - Dual classifier (3class monitor + 4class guidance)
#   4. ours_mon4class  - 4-class sample-level monitoring + spatial CG
#   5. ours_mon3class  - 3-class sample-level monitoring + spatial CG
#   6. safree_dual     - SAFREE + Dual classifier
#   7. safree_mon      - SAFREE + Sample-level monitoring
#   8. sdd             - SDD (fine-tuned UNet)
#   9. esd             - ESD (fine-tuned UNet)
#
# Datasets:
#   ringabell  - nudity-ring-a-bell.csv (79 prompts)
#   i2p        - i2p/sexual.csv (931 prompts)
#   p4dn       - p4dn_16_prompt.csv (16 prompts)
#   unlearndiff - unlearn_diff_nudity.csv
#   mma        - mma-diffusion-nsfw-adv-prompts.csv
#
# Usage:
#   ./generate_all_methods.sh --dataset ringabell --gpu 0
#   ./generate_all_methods.sh --dataset i2p --gpu 0
#   ./generate_all_methods.sh --dataset ringabell --methods "sd_baseline safree" --gpu 0
#   ./generate_all_methods.sh --dataset ringabell --gpu 0 --nohup
#   ./generate_all_methods.sh --dry-run
#
# Best configs should be filled in the BEST CONFIG section below after
# grid search results are aggregated.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/repo_env.sh
source "${SCRIPT_DIR}/lib/repo_env.sh"

# ============================================================================
# Parse Arguments
# ============================================================================
DATASET=""
GPU=0
DRY_RUN=false
USE_NOHUP=false
METHODS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)      DATASET="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --methods)      METHODS_ARG="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --nohup)        USE_NOHUP=true; shift ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Usage: $0 --dataset <ringabell|i2p|p4dn|unlearndiff|mma> --gpu <GPU_ID> [--methods \"method1 method2\"] [--dry-run] [--nohup]"
    echo ""
    echo "Methods: sd_baseline safree ours_dual ours_mon4class ours_mon3class safree_dual safree_mon sdd esd"
    exit 1
fi

# If --nohup, re-launch in background
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${UNLEARNING_REPO_ROOT}/scripts/logs/generate_all_${DATASET}_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    if [ -n "$METHODS_ARG" ]; then
        nohup env CUDA_VISIBLE_DEVICES="$GPU" bash "$0" --dataset "$DATASET" --gpu "$GPU" --methods "$METHODS_ARG" > "$LOG_FILE" 2>&1 &
    else
        nohup env CUDA_VISIBLE_DEVICES="$GPU" bash "$0" --dataset "$DATASET" --gpu "$GPU" > "$LOG_FILE" 2>&1 &
    fi
    echo "PID: $!"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPU

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="${UNLEARNING_REPO_ROOT}"
SCG_DIR="${BASE_DIR}/SoftDelete+CG"
SAFREE_DIR="${BASE_DIR}/SAFREE"
GUIDED2_ROOT="${UNLEARNING_GUIDED2_ROOT}"
PYTHON_BIN="${UNLEARNING_SDD_COPY_PYTHON}"

SD_MODEL="CompVis/stable-diffusion-v1-4"

# SDD / ESD checkpoints
SDD_CKPT="${GUIDED2_ROOT}/Continual2/sdd_2026-01-29_17-05-34"
ESD_CKPT="${GUIDED2_ROOT}/Continual2/esd_2026-01-29_17-05-34"

# Classifiers
CLASSIFIER_3CLASS="${SCG_DIR}/work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLASSIFIER_4CLASS="${SCG_DIR}/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_4CLASS="${SCG_DIR}/gradcam_stats/nudity_4class"
GRADCAM_STATS_3CLASS="${SCG_DIR}/gradcam_stats/nudity_3class"

# Fixed generation params
SEED=42
NUM_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# SAFREE fixed params
SAFREE_ALPHA=0.01
SVF_UP_T=10
CATEGORY="nudity"

# ============================================================================
# Dataset Configuration
# ============================================================================
case "$DATASET" in
    ringabell)
        PROMPT_FILE="${SAFREE_DIR}/datasets/nudity-ring-a-bell.csv"
        OUTPUT_BASE="${SCG_DIR}/scg_outputs/final_${DATASET}"
        SAFREE_CONCEPT="sexual"
        ;;
    i2p)
        PROMPT_FILE="${GUIDED2_ROOT}/prompts/i2p/sexual.csv"
        OUTPUT_BASE="${SCG_DIR}/scg_outputs/final_${DATASET}"
        SAFREE_CONCEPT="sexual"
        ;;
    p4dn)
        PROMPT_FILE="${SAFREE_DIR}/datasets/p4dn_16_prompt.csv"
        OUTPUT_BASE="${SCG_DIR}/scg_outputs/final_${DATASET}"
        SAFREE_CONCEPT="sexual"
        ;;
    unlearndiff)
        PROMPT_FILE="${SAFREE_DIR}/datasets/unlearn_diff_nudity.csv"
        OUTPUT_BASE="${SCG_DIR}/scg_outputs/final_${DATASET}"
        SAFREE_CONCEPT="sexual"
        ;;
    mma)
        PROMPT_FILE="${SAFREE_DIR}/datasets/mma-diffusion-nsfw-adv-prompts.csv"
        OUTPUT_BASE="${SCG_DIR}/scg_outputs/final_${DATASET}"
        SAFREE_CONCEPT="sexual"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: ringabell, i2p, p4dn, unlearndiff, mma"
        exit 1
        ;;
esac

# ============================================================================
# BEST CONFIG (fill in after grid search)
# ============================================================================
# Format: GUIDANCE_SCALE BASE_GUIDANCE_SCALE SPATIAL_START SPATIAL_END [extras]

# --- Ours: Dual Classifier --- (gs17.5_hs1.5_bs1.0_sp0.3-0.7)
DUAL_GS=17.5
DUAL_HS=1.5
DUAL_BASE_GS=1.0
DUAL_SP_START=0.3
DUAL_SP_END=0.7

# --- Ours: Monitor 4class --- (best: mon0.1_gs12.5_sp0.1-0.4_bs1.0)
MON4_GS=12.5
MON4_BASE_GS=1.0
MON4_SP_START=0.1
MON4_SP_END=0.4
MON4_THRESHOLD=0.1

# --- Ours: Monitor 3class --- (best: mon0.1_gs12.5_sp0.1-0.4_bs1.0)
MON3_GS=12.5
MON3_BASE_GS=1.0
MON3_SP_START=0.1
MON3_SP_END=0.4
MON3_THRESHOLD=0.1

# --- SAFREE + Dual --- (best: gs7.5_hs1.5_bs2.0_sp0.3-0.3, ringabell SR=89.9%)
SFDUAL_GS=7.5
SFDUAL_HS=1.5
SFDUAL_BASE_GS=2.0
SFDUAL_SP_START=0.3
SFDUAL_SP_END=0.3

# --- SAFREE + Monitor --- (best: mon0.3_gs12.5_bs2.0_sp0.5-0.5, ringabell SR=89.9%)
SFMON_GS=12.5
SFMON_BASE_GS=2.0
SFMON_SP_START=0.5
SFMON_SP_END=0.5
SFMON_THRESHOLD=0.3
SFMON_HS=1.0

# ============================================================================
# Methods
# ============================================================================
if [ -n "$METHODS_ARG" ]; then
    METHODS=($METHODS_ARG)
else
    METHODS=(sd_baseline safree ours_dual ours_mon4class ours_mon3class safree_dual safree_mon sdd esd)
fi

mkdir -p "${OUTPUT_BASE}/logs"

echo "============================================================"
echo "Unified Generation: ${DATASET}"
echo "============================================================"
echo "GPU: ${GPU}"
echo "Prompt: ${PROMPT_FILE}"
echo "Output: ${OUTPUT_BASE}"
echo "Methods: ${METHODS[*]}"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE]"
    echo ""
fi

# ============================================================================
# Generation Functions
# ============================================================================

generate_sd_baseline() {
    local output_dir="${OUTPUT_BASE}/sd_baseline"
    local log_file="${OUTPUT_BASE}/logs/sd_baseline.log"
    echo "[SD Baseline] → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SCG_DIR}"

    "${PYTHON_BIN}" -c "
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import csv

# Load prompts
prompts = []
with open('${PROMPT_FILE}', 'r') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    col_priority = ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']
    prompt_col = next((c for c in col_priority if c in fields), None)
    if not prompt_col:
        raise ValueError(f'No prompt column found. Available: {fields}')
    print(f'Using column: {prompt_col}')
    for row in reader:
        val = row.get(prompt_col, '').strip()
        if val:
            prompts.append(val)

print(f'Loaded {len(prompts)} prompts')

pipe = StableDiffusionPipeline.from_pretrained(
    '${SD_MODEL}',
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to('cuda')

output_dir = Path('${output_dir}')
output_dir.mkdir(parents=True, exist_ok=True)

generator = torch.Generator(device='cuda').manual_seed(${SEED})

for i, prompt in enumerate(prompts):
    print(f'[{i+1}/{len(prompts)}] {prompt[:50]}...')
    image = pipe(
        prompt,
        num_inference_steps=${NUM_STEPS},
        guidance_scale=${CFG_SCALE},
        generator=generator
    ).images[0]
    image.save(output_dir / f'{i:06d}.png')
    generator = torch.Generator(device='cuda').manual_seed(${SEED})

print(f'Done: {len(prompts)} images')
" > "$log_file" 2>&1

    echo "[SD Baseline] Done"
}

generate_safree() {
    local output_dir="${OUTPUT_BASE}/safree"
    local log_file="${OUTPUT_BASE}/logs/safree.log"
    echo "[SAFREE] → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SAFREE_DIR}"

    "${PYTHON_BIN}" gen_safree_i2p_concepts.py \
        --prompt_file "${PROMPT_FILE}" \
        --concepts "${SAFREE_CONCEPT}" \
        --model_id "${SD_MODEL}" \
        --outdir "${output_dir}" \
        --num_images ${NSAMPLES} \
        --steps ${NUM_STEPS} \
        --guidance ${CFG_SCALE} \
        --seed ${SEED} \
        --safree \
        --svf \
        --up_t ${SVF_UP_T} \
        --device "cuda:0" \
        --no_concept_subdir \
        > "$log_file" 2>&1

    echo "[SAFREE] Done"
}

generate_ours_dual() {
    local output_dir="${OUTPUT_BASE}/ours_dual"
    local log_file="${OUTPUT_BASE}/logs/ours_dual.log"
    echo "[Ours-Dual] gs=${DUAL_GS} hs=${DUAL_HS} bs=${DUAL_BASE_GS} sp=${DUAL_SP_START}-${DUAL_SP_END}"
    echo "  → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SCG_DIR}"

    "${PYTHON_BIN}" generate_nudity_dual_classifier.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_3class_ckpt "${CLASSIFIER_3CLASS}" \
        --classifier_4class_ckpt "${CLASSIFIER_4CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_4CLASS}" \
        --guidance_scale ${DUAL_GS} \
        --harmful_scale ${DUAL_HS} \
        --base_guidance_scale ${DUAL_BASE_GS} \
        --spatial_threshold_start ${DUAL_SP_START} \
        --spatial_threshold_end ${DUAL_SP_END} \
        --spatial_threshold_strategy cosine \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "$log_file" 2>&1

    echo "[Ours-Dual] Done"
}

generate_ours_mon4class() {
    local output_dir="${OUTPUT_BASE}/ours_mon4class"
    local log_file="${OUTPUT_BASE}/logs/ours_mon4class.log"
    echo "[Ours-Mon4class] gs=${MON4_GS} bs=${MON4_BASE_GS} sp=${MON4_SP_START}-${MON4_SP_END} thr=${MON4_THRESHOLD}"
    echo "  → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SCG_DIR}"

    "${PYTHON_BIN}" generate_nudity_4class_sample_level_monitoring.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_4CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_4CLASS}" \
        --monitoring_threshold ${MON4_THRESHOLD} \
        --guidance_scale ${MON4_GS} \
        --base_guidance_scale ${MON4_BASE_GS} \
        --spatial_threshold_start ${MON4_SP_START} \
        --spatial_threshold_end ${MON4_SP_END} \
        --spatial_threshold_strategy cosine \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "$log_file" 2>&1

    echo "[Ours-Mon4class] Done"
}

generate_ours_mon3class() {
    local output_dir="${OUTPUT_BASE}/ours_mon3class"
    local log_file="${OUTPUT_BASE}/logs/ours_mon3class.log"
    echo "[Ours-Mon3class] gs=${MON3_GS} bs=${MON3_BASE_GS} sp=${MON3_SP_START}-${MON3_SP_END} thr=${MON3_THRESHOLD}"
    echo "  → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SCG_DIR}"

    "${PYTHON_BIN}" generate_nudity_3class_sample_level_monitoring.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_3CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_3CLASS}" \
        --monitoring_threshold ${MON3_THRESHOLD} \
        --guidance_scale ${MON3_GS} \
        --base_guidance_scale ${MON3_BASE_GS} \
        --spatial_threshold_start ${MON3_SP_START} \
        --spatial_threshold_end ${MON3_SP_END} \
        --spatial_threshold_strategy cosine \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "$log_file" 2>&1

    echo "[Ours-Mon3class] Done"
}

generate_safree_dual() {
    local output_dir="${OUTPUT_BASE}/safree_dual"
    local log_file="${OUTPUT_BASE}/logs/safree_dual.log"
    echo "[SAFREE+Dual] gs=${SFDUAL_GS} hs=${SFDUAL_HS} bs=${SFDUAL_BASE_GS} sp=${SFDUAL_SP_START}-${SFDUAL_SP_END}"
    echo "  → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SAFREE_DIR}"

    "${PYTHON_BIN}" generate_safree_dual_classifier.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_3class_ckpt "${CLASSIFIER_3CLASS}" \
        --classifier_4class_ckpt "${CLASSIFIER_4CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_4CLASS}" \
        --safree \
        --safree_alpha ${SAFREE_ALPHA} \
        --svf \
        --svf_up_t ${SVF_UP_T} \
        --category "${CATEGORY}" \
        --guidance_scale ${SFDUAL_GS} \
        --harmful_scale ${SFDUAL_HS} \
        --base_guidance_scale ${SFDUAL_BASE_GS} \
        --spatial_threshold_start ${SFDUAL_SP_START} \
        --spatial_threshold_end ${SFDUAL_SP_END} \
        --spatial_threshold_strategy cosine \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "$log_file" 2>&1

    echo "[SAFREE+Dual] Done"
}

generate_safree_mon() {
    local output_dir="${OUTPUT_BASE}/safree_mon"
    local log_file="${OUTPUT_BASE}/logs/safree_mon.log"
    echo "[SAFREE+Mon] gs=${SFMON_GS} bs=${SFMON_BASE_GS} sp=${SFMON_SP_START}-${SFMON_SP_END} thr=${SFMON_THRESHOLD}"
    echo "  → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${SAFREE_DIR}"

    "${PYTHON_BIN}" generate_safree_monitoring.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_4CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_4CLASS}" \
        --safree \
        --safree_alpha ${SAFREE_ALPHA} \
        --svf \
        --svf_up_t ${SVF_UP_T} \
        --category "${CATEGORY}" \
        --monitoring_threshold ${SFMON_THRESHOLD} \
        --guidance_scale ${SFMON_GS} \
        --harmful_scale ${SFMON_HS} \
        --base_guidance_scale ${SFMON_BASE_GS} \
        --spatial_threshold_start ${SFMON_SP_START} \
        --spatial_threshold_end ${SFMON_SP_END} \
        --spatial_threshold_strategy cosine \
        --num_inference_steps ${NUM_STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "$log_file" 2>&1

    echo "[SAFREE+Mon] Done"
}

generate_sdd() {
    local output_dir="${OUTPUT_BASE}/sdd"
    local log_file="${OUTPUT_BASE}/logs/sdd.log"
    echo "[SDD] → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${GUIDED2_ROOT}"

    "${PYTHON_BIN}" generate.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --unet_path "${SDD_CKPT}/step=001000" \
        --image_dir "${output_dir}" \
        --prompt_path "${PROMPT_FILE}" \
        --num_images_per_prompt ${NSAMPLES} \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --device "cuda:0" \
        > "$log_file" 2>&1

    echo "[SDD] Done"
}

generate_esd() {
    local output_dir="${OUTPUT_BASE}/esd"
    local log_file="${OUTPUT_BASE}/logs/esd.log"
    echo "[ESD] → ${output_dir}"

    [ "$DRY_RUN" = true ] && return 0
    mkdir -p "$output_dir"

    cd "${GUIDED2_ROOT}"

    "${PYTHON_BIN}" generate.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --unet_path "${ESD_CKPT}/step=001000" \
        --image_dir "${output_dir}" \
        --prompt_path "${PROMPT_FILE}" \
        --num_images_per_prompt ${NSAMPLES} \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --device "cuda:0" \
        > "$log_file" 2>&1

    echo "[ESD] Done"
}

# ============================================================================
# Run Selected Methods
# ============================================================================
for method in "${METHODS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    case "$method" in
        sd_baseline)    generate_sd_baseline ;;
        safree)         generate_safree ;;
        ours_dual)      generate_ours_dual ;;
        ours_mon4class) generate_ours_mon4class ;;
        ours_mon3class) generate_ours_mon3class ;;
        safree_dual)    generate_safree_dual ;;
        safree_mon)     generate_safree_mon ;;
        sdd)            generate_sdd ;;
        esd)            generate_esd ;;
        *)              echo "Unknown method: $method"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "ALL GENERATION COMPLETE"
echo "============================================================"
echo "Output: ${OUTPUT_BASE}"
echo "Logs:   ${OUTPUT_BASE}/logs/"
echo ""
echo "Next: Run VLM evaluation"
echo "  CUDA_VISIBLE_DEVICES=0 bash vlm/batch_eval_qwen_multi_gpu.sh ${OUTPUT_BASE}"
echo "============================================================"
