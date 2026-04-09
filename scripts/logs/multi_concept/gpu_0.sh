#!/usr/bin/env bash
# Auto-generated GPU 0 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=0"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 0 starting"

# --- Job: sexual / v19_hybrid (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sexual/v19_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_hybrid_ss3.0_st0.3 --cas_threshold 0.6 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked nudity nsfw bare body --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: sexual/v19_hybrid"
fi

# --- Job: compound_all7 / v14_all7 (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: compound_all7/v14_all7"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v14_dag_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v14_dag_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v14_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked blood weapon fight bully threat hate gore drug suicide --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/hate /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/shocking /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/illegal_activity /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/self-harm --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: compound_all7/v14_all7"
fi

# --- Job: self-harm / v19_hybrid (est: 280 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: self-harm/v19_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_hybrid_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words suicide cut self-harm wound bleed --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/self-harm --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: self-harm/v19_hybrid"
fi

# --- Job: violence / v14_dag_adaptive (est: 264 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: violence/v14_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words blood wound weapon fight kill gore injury --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: violence/v14_dag_adaptive"
fi

# --- Job: harassment / sd_baseline (est: 109 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: harassment/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_harassment.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: harassment/sd_baseline"
fi

# --- Job: illegal_activity / sd_baseline (est: 96 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: illegal_activity/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: illegal_activity/sd_baseline"
fi

# --- Job: hate / safree (est: 81 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: hate/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_hate.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: hate/safree"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 0 ALL DONE (7 jobs)"
