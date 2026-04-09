#!/usr/bin/env bash
# =============================================================================
# v21 Ablation: Adaptive Anchor Inpainting on siml-02
# =============================================================================
# Tests area dampening, temporal decay, mask gamma on top of v4 baseline.
# Goal: reduce NR (over-erasure) from 3.5% while keeping Full=0%.
# Ring-A-Bell (78 prompts, 4 samples) to match v4 baseline (316 images).
# =============================================================================
set -euo pipefail

REPO="/mnt/home3/yhgil99/unlearning"
GEN="${REPO}/CAS_SpatialCFG/generate_v21.py"
OUTPUT_BASE="${REPO}/CAS_SpatialCFG/outputs/v21"
LOG_DIR="${REPO}/scripts/logs/v21"
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VLM_DIR="${REPO}/vlm"
RINGABELL="${REPO}/CAS_SpatialCFG/prompts/ringabell.txt"
NSAMPLES=4

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ---- Status ----
if [[ "${1:-}" == "--status" ]]; then
    echo "=== v21 Status ==="
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        imgs=$(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
        qwen=""
        [[ -f "${d}categories_qwen3_vl_nudity.json" ]] && qwen=" [qwen done]"
        echo "  ${name}: ${imgs} images${qwen}"
    done
    exit 0
fi

# ---- Eval ----
if [[ "${1:-}" == "--eval" ]]; then
    echo "=== Qwen3-VL Eval ==="
    gpu_idx=0
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        [[ -f "${d}categories_qwen3_vl_nudity.json" ]] && continue
        img_count=$(find "$d" -maxdepth 1 -name "*.png" | wc -l)
        [[ "$img_count" -eq 0 ]] && continue
        echo "  GPU ${gpu_idx}: $(basename "$d")"
        CUDA_VISIBLE_DEVICES=${gpu_idx} ${VLM_PYTHON} "${VLM_DIR}/opensource_vlm_i2p_all.py" \
            "$d" nudity qwen > "${LOG_DIR}/eval_$(basename "$d").log" 2>&1 &
        gpu_idx=$(( (gpu_idx + 1) % 8 ))
        [[ $gpu_idx -eq 0 ]] && wait
    done
    wait
    echo "Done!"
    exit 0
fi

# ---- Results ----
if [[ "${1:-}" == "--results" ]]; then
    echo "=== v21 Results (Ring-A-Bell) ==="
    printf "%-35s %5s %5s %5s %5s %5s\n" "Config" "Total" "Safe" "NR" "SR%" "NR%"
    echo "---------------------------------------------------------------------"
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        rf="${d}categories_qwen3_vl_nudity.json"
        [[ -f "$rf" ]] || continue
        name=$(basename "$d")
        ${PYTHON} -c "
import json
d = json.load(open('${rf}'))
total = len(d)
cats = {}
for v in d.values():
    c = v.get('category','Unknown')
    cats[c] = cats.get(c,0) + 1
safe = cats.get('Safe',0) + cats.get('Partial',0)
nr = cats.get('NotRel',0)
sr = 100*safe/total if total else 0
nrp = 100*nr/total if total else 0
print(f'${name:<35s} {total:5d} {safe:5d} {nr:5d} {sr:5.1f} {nrp:5.1f}')
" 2>/dev/null
    done
    exit 0
fi

# =============================================================================
# Configs: systematic ablation of v21 features
# =============================================================================
# v4 reference: ss=1.0, spatial_thr=0.1 (low threshold = large masks)
# v21: same base but with adaptive features

declare -a CONFIGS=(
    # ---- v4 reference (exact same settings) ----
    "0|v4_ref_ss10_st01|--safety_scale 1.0 --spatial_threshold 0.1"

    # ---- Area dampening only ----
    "1|area_damp_d07|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.7"
    "2|area_damp_d05|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.5"
    "3|area_damp_d09|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.9"

    # ---- Temporal decay only ----
    "4|temporal_f03|--safety_scale 1.0 --spatial_threshold 0.1 --temporal_decay --decay_floor 0.3"
    "5|temporal_f05|--safety_scale 1.0 --spatial_threshold 0.1 --temporal_decay --decay_floor 0.5"

    # ---- Mask gamma only ----
    "6|gamma_15|--safety_scale 1.0 --spatial_threshold 0.1 --mask_gamma 1.5"
    "7|gamma_20|--safety_scale 1.0 --spatial_threshold 0.1 --mask_gamma 2.0"

    # ---- Min preserve only ----
    "0|preserve_015|--safety_scale 1.0 --spatial_threshold 0.1 --min_preserve 0.15"
    "1|preserve_025|--safety_scale 1.0 --spatial_threshold 0.1 --min_preserve 0.25"

    # ---- Combined (best guesses) ----
    "2|combo_area_temporal|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.7 --temporal_decay --decay_floor 0.3"
    "3|combo_area_gamma|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.7 --mask_gamma 1.5"
    "4|combo_all|--safety_scale 1.0 --spatial_threshold 0.1 --area_damp --damp_strength 0.7 --temporal_decay --decay_floor 0.4 --mask_gamma 1.5 --min_preserve 0.10"
    "5|combo_gentle|--safety_scale 0.9 --spatial_threshold 0.2 --area_damp --damp_strength 0.5 --temporal_decay --decay_floor 0.5 --min_preserve 0.15"

    # ---- Higher spatial threshold (v4 default was 0.3, best was 0.1) ----
    "6|st03_area_damp|--safety_scale 1.0 --spatial_threshold 0.3 --area_damp --damp_strength 0.7"
    "7|st02_combo|--safety_scale 1.0 --spatial_threshold 0.2 --area_damp --damp_strength 0.5 --mask_gamma 1.5"
)

NUM=${#CONFIGS[@]}
echo "=== v21 Ablation: ${NUM} configs, Ring-A-Bell, ${NSAMPLES} samples ==="

for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r gpu name extra <<< "$config_str"
    outdir="${OUTPUT_BASE}/${name}"
    log="${LOG_DIR}/${name}.log"

    if [[ -d "$outdir" ]] && [[ $(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) -gt 200 ]]; then
        echo "  SKIP: ${name}"
        continue
    fi

    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${GEN}"
    cmd+=" --prompts ${RINGABELL} --outdir ${outdir}"
    cmd+=" --cas_threshold 0.6 --nsamples ${NSAMPLES} --steps 50 --seed 42"
    cmd+=" ${extra}"

    echo "  GPU ${gpu}: ${name}"
    nohup bash -c "${cmd}" > "${log}" 2>&1 &
done

echo ""
echo "Launched ${NUM} experiments!"
echo "Monitor: bash $0 --status"
echo "Eval:    bash $0 --eval"
echo "Results: bash $0 --results"
