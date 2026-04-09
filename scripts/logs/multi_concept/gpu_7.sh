#!/usr/bin/env bash
# Auto-generated GPU 7 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=7"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 7 starting"

# --- Job: compound_all7 / v19_all7 (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: compound_all7/v19_all7"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v19_dag_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v19_dag_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/all_7_concepts/v19_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked blood weapon fight bully threat hate gore drug suicide --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/hate /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/shocking /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/illegal_activity /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/self-harm --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: compound_all7/v19_all7"
fi

# --- Job: shocking / safree (est: 299 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: shocking/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_shocking.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: shocking/safree"
fi

# --- Job: harassment / safree (est: 288 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: harassment/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_harassment.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: harassment/safree"
fi

# --- Job: violence / v14_hybrid (est: 264 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: violence/v14_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v14_hybrid_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words blood wound weapon fight kill gore injury --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: violence/v14_hybrid"
fi

# --- Job: illegal_activity / v14_dag_adaptive (est: 254 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: illegal_activity/v14_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words drug steal crime illegal weapon --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/illegal_activity --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: illegal_activity/v14_dag_adaptive"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 7 ALL DONE (5 jobs)"
