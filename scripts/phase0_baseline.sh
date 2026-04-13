#!/bin/bash
# Phase 0-B: Generate SD v1.4 baseline images for all datasets
# Run on siml-01 GPU 3-7
set -e

cd /mnt/home3/yhgil99/unlearning

CONDA_ENV="sdd_copy"
GEN_SCRIPT="/mnt/home3/yhgil99/guided2-safe-diffusion/generate.py"
BASELINE_DIR="CAS_SpatialCFG/outputs/baselines_v2"

source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV

mkdir -p $BASELINE_DIR

gen_baseline() {
    local name=$1
    local prompt_path=$2
    local gpu=$3
    local out_dir="${BASELINE_DIR}/${name}"

    if [ -d "$out_dir" ] && [ "$(ls -1 $out_dir/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[SKIP] $name already has images"
        return
    fi

    echo "[GPU $gpu] Baseline: $name ($prompt_path)"
    CUDA_VISIBLE_DEVICES=$gpu python3 $GEN_SCRIPT \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --image_dir "$out_dir" \
        --prompt_path "$prompt_path" \
        --num_images_per_prompt 1 \
        --num_inference_steps 50 \
        --seed 42 \
        --device "cuda:0" \
        2>&1 | tee "${out_dir}_log.txt"
}

# Nudity datasets
gen_baseline "rab"          "CAS_SpatialCFG/prompts/ringabell.txt"     3 &
gen_baseline "mma"          "CAS_SpatialCFG/prompts/mma.txt"           4 &
gen_baseline "p4dn"         "CAS_SpatialCFG/prompts/p4dn.txt"         5 &
gen_baseline "unlearndiff"  "CAS_SpatialCFG/prompts/unlearndiff.txt"   5 &
wait

# MJA datasets
gen_baseline "mja_sexual"     "CAS_SpatialCFG/prompts/mja_sexual.txt"     6 &
gen_baseline "mja_violent"    "CAS_SpatialCFG/prompts/mja_violent.txt"     6 &
gen_baseline "mja_disturbing" "CAS_SpatialCFG/prompts/mja_disturbing.txt"  7 &
gen_baseline "mja_illegal"    "CAS_SpatialCFG/prompts/mja_illegal.txt"     7 &
wait

echo "=== All baseline generation complete ==="
