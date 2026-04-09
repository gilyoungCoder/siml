#!/usr/bin/env bash
# Auto-generated GPU 2 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=2"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 2 starting"

# --- Job: sexual / v14_hybrid (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sexual/v14_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v14_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v14_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v14_hybrid_ss3.0_st0.3 --cas_threshold 0.6 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked nudity nsfw bare body --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: sexual/v14_hybrid"
fi

# --- Job: compound_3way / v14_compound3 (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: compound_3way/v14_compound3"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v14_dag_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v14_dag_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v14_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked blood weapon fight bully threat --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: compound_3way/v14_compound3"
fi

# --- Job: self-harm / v14_hybrid (est: 280 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: self-harm/v14_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v14_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v14_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v14_hybrid_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words suicide cut self-harm wound bleed --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/self-harm --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: self-harm/v14_hybrid"
fi

# --- Job: illegal_activity / v19_hybrid (est: 254 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: illegal_activity/v19_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v19_text_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v19_text_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v19_text_hybrid_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words drug steal crime illegal weapon --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/illegal_activity --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: illegal_activity/v19_hybrid"
fi

# --- Job: illegal_activity / safree (est: 254 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: illegal_activity/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: illegal_activity/safree"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 2 ALL DONE (5 jobs)"
