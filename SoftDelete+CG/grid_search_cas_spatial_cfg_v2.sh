#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Grid Search v2: CAS + Spatial CFG (weighted_cas focus)
# GPU 0-6 (GPU 7 reserved)
# =============================================================================

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home/yhgil99/.conda/envs/sdd/bin/python"
SCRIPT_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG"
CKPT="CompVis/stable-diffusion-v1-4"
SEXUAL_FILE="${SCRIPT_DIR}/prompts/sexual.csv"
COCO_FILE="${SCRIPT_DIR}/prompts/coco_30.txt"
BASE_OUTPUT="${SCRIPT_DIR}/scg_outputs/cas_spatial_cfg_v2"

NSAMPLES=4
STEPS=50
SEED=42
CFG=7.5

# Common good settings from initial tests
CAS_THRESHOLD=0.3
SPATIAL_METHOD="weighted_cas"
BLUR=2.0

mkdir -p "${BASE_OUTPUT}"

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
        --target_concepts "nudity" "nude person" "nsfw person" "naked body" \
        --anchor_concepts "clothed person" "person wearing clothes" "fully dressed person" \
        --save_spatial_maps \
        --debug \
        "${extra_args[@]}" \
        > "${log}" 2>&1 &
    echo "[GPU ${gpu}] PID: $! → ${name}"
}

echo "============================================================"
echo "CAS + Spatial CFG Grid Search v2 (weighted_cas)"
echo "============================================================"

# =============================================================================
# Batch 1: Safety scale + guidance mode sweep
# =============================================================================

# GPU 0: weighted_cas + anchor_shift, scale 1.0 (baseline)
run_exp 0 "wcas_ashift_s1.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 1: anchor_shift, scale 2.0 (stronger)
run_exp 1 "wcas_ashift_s2.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 2.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 2: anchor_shift, scale 3.0 (very strong)
run_exp 2 "wcas_ashift_s3.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 3.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 3: target_negate mode
run_exp 3 "wcas_tneg_s1.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode target_negate --safety_scale 1.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 4: dual mode
run_exp 4 "wcas_dual_s1.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode dual --safety_scale 1.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 5: target_negate, scale 2.0
run_exp 5 "wcas_tneg_s2.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode target_negate --safety_scale 2.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 6: dual mode, scale 2.0
run_exp 6 "wcas_dual_s2.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode dual --safety_scale 2.0 \
    --guidance_schedule constant --warmup_steps 3

echo ""
echo "Batch 1: 7 experiments launched, waiting..."
wait
echo "Batch 1 done!"

# =============================================================================
# Batch 2: CAS threshold + spatial threshold + schedule + COCO
# =============================================================================

# GPU 0: CAS threshold 0.2 (more sensitive)
run_exp 0 "wcas_cas02" "${SEXUAL_FILE}" \
    --cas_threshold 0.2 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 1: CAS threshold 0.5 (more selective)
run_exp 1 "wcas_cas05" "${SEXUAL_FILE}" \
    --cas_threshold 0.5 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 2: Spatial threshold 0.2 (wider guidance area)
run_exp 2 "wcas_sthr02" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.2 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 3: Spatial threshold 0.5 (tighter guidance area)
run_exp 3 "wcas_sthr05" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.5 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 4: Cosine decay schedule
run_exp 4 "wcas_cosine_s2" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 2.0 \
    --guidance_schedule cosine --warmup_steps 3

# GPU 5: COCO benign — false positive check
run_exp 5 "wcas_COCO_s1.5" "${COCO_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 6: No sticky + scale 1.5
run_exp 6 "wcas_nosticky_s1.5" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_no_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

echo ""
echo "Batch 2: 7 experiments launched, waiting..."
wait
echo "Batch 2 done!"

# =============================================================================
# Batch 3: spatial_cas comparison + high scale experiments
# =============================================================================

# GPU 0: spatial_cas method (comparison)
run_exp 0 "scas_ashift_s1.5" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method spatial_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 1: weighted_cas + high spatial_scale_high (10)
run_exp 1 "wcas_shigh10_s1.5" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 10.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 2: weighted_cas + non-zero spatial_scale_low (gentle global push)
run_exp 2 "wcas_slow1_s1.5" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 1.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 3: weighted_cas + larger blur sigma
run_exp 3 "wcas_blur4_s1.5" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma 4.0 --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

# GPU 4: anchor_shift scale 5.0 (very strong)
run_exp 4 "wcas_ashift_s5.0" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 5.0 \
    --guidance_schedule constant --warmup_steps 3

# GPU 5: linear_decay schedule with scale 3.0
run_exp 5 "wcas_lindecay_s3" "${SEXUAL_FILE}" \
    --cas_threshold 0.3 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 3.0 \
    --guidance_schedule linear_decay --warmup_steps 3

# GPU 6: COCO with stricter CAS (false positive test)
run_exp 6 "wcas_COCO_cas05" "${COCO_FILE}" \
    --cas_threshold 0.5 --cas_sticky \
    --spatial_method weighted_cas --spatial_threshold 0.3 \
    --spatial_scale_high 5.0 --spatial_scale_low 0.0 \
    --mask_blur_sigma ${BLUR} --adaptive_area_scale \
    --guidance_mode anchor_shift --safety_scale 1.5 \
    --guidance_schedule constant --warmup_steps 3

echo ""
echo "Batch 3: 7 experiments launched, waiting..."
wait
echo "Batch 3 done!"

echo ""
echo "============================================================"
echo "ALL 21 EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""

# Summary
echo "--- Results Summary ---"
for dir in ${BASE_OUTPUT}/*/; do
    name=$(basename "$dir")
    stats="${dir}generation_stats.json"
    if [ -f "$stats" ]; then
        result=$(${PYTHON} -c "
import json
with open('${stats}') as f:
    d = json.load(f)
o = d.get('overall', {})
imgs = d.get('per_image_stats', [])
guided_areas = []
for img in imgs:
    for s in img.get('step_history', []):
        if s.get('guided') and 'spatial_area' in s:
            guided_areas.append(s['spatial_area'])
avg_area = sum(guided_areas)/len(guided_areas) if guided_areas else 0
print(f\"trig={o.get('triggered_count','?')}/{o.get('total_images','?')} cas={o.get('avg_cas_score',0):.3f} guided={o.get('avg_guided_steps',0):.1f} area={avg_area:.3f}\")
" 2>/dev/null || echo "parse error")
        printf "  %-30s %s\n" "${name}" "${result}"
    fi
done
