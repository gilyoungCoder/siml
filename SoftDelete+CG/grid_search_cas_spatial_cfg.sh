#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Grid Search: CAS + Spatial CFG
# Run multiple variations across GPUs 0-6 (GPU 7 reserved)
# =============================================================================

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home/yhgil99/.conda/envs/sdd/bin/python"
SCRIPT_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG"
CKPT="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="${SCRIPT_DIR}/prompts/sexual.csv"
COCO_FILE="${SCRIPT_DIR}/prompts/coco_30.txt"
BASE_OUTPUT="${SCRIPT_DIR}/scg_outputs/cas_spatial_cfg"

NSAMPLES=4
STEPS=50
SEED=42
CFG=7.5

mkdir -p "${BASE_OUTPUT}"

# Function to run experiment in background
run_exp() {
    local gpu=$1
    local name=$2
    local prompt=$3
    shift 3
    local extra_args=("$@")

    local output_dir="${BASE_OUTPUT}/${name}"
    local log="${output_dir}/run.log"
    mkdir -p "${output_dir}"

    echo "[GPU ${gpu}] Starting: ${name}"
    CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${SCRIPT_DIR}/generate_cas_spatial_cfg.py \
        --ckpt_path "${CKPT}" \
        --prompt_file "${prompt}" \
        --output_dir "${output_dir}" \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG} \
        --num_inference_steps ${STEPS} \
        --seed ${SEED} \
        --save_spatial_maps \
        --debug \
        "${extra_args[@]}" \
        > "${log}" 2>&1 &
    echo "[GPU ${gpu}] PID: $! → ${name}"
}

echo "============================================================"
echo "CAS + Spatial CFG Grid Search"
echo "============================================================"
echo ""

# =============================================================================
# Batch 1: Spatial method comparison + CAS threshold sweep (GPU 0-6)
# =============================================================================

# --- GPU 0: target_strength + anchor_shift (baseline) ---
run_exp 0 "v1_tstr_ashift_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 1: cosine_diff + anchor_shift ---
run_exp 1 "v2_cosdiff_ashift_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "cosine_diff" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 2: diff_norm + anchor_shift ---
run_exp 2 "v3_diffnorm_ashift_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "diff_norm" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 3: target_projection + anchor_shift ---
run_exp 3 "v4_tproj_ashift_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_projection" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 4: target_strength + target_negate (SLD-style) ---
run_exp 4 "v5_tstr_tnegate_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "target_negate" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 5: target_strength + dual mode ---
run_exp 5 "v6_tstr_dual_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "dual" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 6: Higher safety scale (2.0) + target_strength ---
run_exp 6 "v7_tstr_ashift_scale2_cas03" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 2.0 \
    --guidance_schedule "constant" --warmup_steps 3

echo ""
echo "============================================================"
echo "Batch 1 launched on GPUs 0-6 (7 experiments)"
echo "Monitor: tail -f ${BASE_OUTPUT}/v*/run.log"
echo "============================================================"
echo ""
echo "Waiting for Batch 1 to complete..."
wait

echo ""
echo "============================================================"
echo "Batch 1 COMPLETE! Starting Batch 2..."
echo "============================================================"
echo ""

# =============================================================================
# Batch 2: CAS threshold sweep + COCO benign test (GPU 0-6)
# =============================================================================

# --- GPU 0: CAS threshold 0.1 (very sensitive) ---
run_exp 0 "v8_tstr_ashift_cas01" "${PROMPT_FILE}" \
    --cas_threshold 0.1 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 1: CAS threshold 0.5 ---
run_exp 1 "v9_tstr_ashift_cas05" "${PROMPT_FILE}" \
    --cas_threshold 0.5 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 2: CAS threshold 0.2 ---
run_exp 2 "v10_tstr_ashift_cas02" "${PROMPT_FILE}" \
    --cas_threshold 0.2 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 3: COCO benign (false positive check) with CAS 0.3 ---
run_exp 3 "v11_coco_tstr_cas03" "${COCO_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 4: Higher spatial scale (10.0) ---
run_exp 4 "v12_tstr_ashift_sscale10" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 10.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

# --- GPU 5: Cosine schedule ---
run_exp 5 "v13_tstr_ashift_cosine_sched" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.5 \
    --guidance_schedule "cosine" --warmup_steps 3

# --- GPU 6: No sticky CAS (re-evaluate every step) ---
run_exp 6 "v14_tstr_ashift_nosticky" "${PROMPT_FILE}" \
    --cas_threshold 0.3 --cas_no_sticky \
    --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
    --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
    --spatial_method "target_strength" \
    --spatial_threshold 0.3 --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 1.0 --adaptive_area_scale \
    --guidance_mode "anchor_shift" --safety_scale 1.0 \
    --guidance_schedule "constant" --warmup_steps 3

echo ""
echo "============================================================"
echo "Batch 2 launched on GPUs 0-6 (7 experiments)"
echo "Waiting for completion..."
echo "============================================================"
wait

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================================"
echo "Results at: ${BASE_OUTPUT}/"
echo ""

# Print summary of all experiments
echo "--- Summary ---"
for dir in ${BASE_OUTPUT}/v*/; do
    name=$(basename "$dir")
    stats="${dir}generation_stats.json"
    if [ -f "$stats" ]; then
        triggered=$(${PYTHON} -c "
import json
with open('${stats}') as f:
    d = json.load(f)
o = d.get('overall', {})
print(f\"triggered={o.get('triggered_count','?')}/{o.get('total_images','?')} avg_cas={o.get('avg_cas_score',0):.4f} avg_guided={o.get('avg_guided_steps',0):.1f}\")
" 2>/dev/null || echo "parse error")
        echo "  ${name}: ${triggered}"
    fi
done
