#!/usr/bin/env bash
# Auto-generated GPU 1 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=1"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 1 starting"

# --- Job: sexual / v19_dag_adaptive (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sexual/v19_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/v19_text_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.6 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked nudity nsfw bare body --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: sexual/v19_dag_adaptive"
fi

# --- Job: compound_3way / v19_compound3 (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: compound_3way/v19_compound3"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v19_dag_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v19_dag_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sex_vio_harass/v19_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked blood weapon fight bully threat --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: compound_3way/v19_compound3"
fi

# --- Job: self-harm / v19_dag_adaptive (est: 280 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: self-harm/v19_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/v19_text_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words suicide cut self-harm wound bleed --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/self-harm --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: self-harm/v19_dag_adaptive"
fi

# --- Job: violence / safree (est: 264 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: violence/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: violence/safree"
fi

# --- Job: self-harm / sd_baseline (est: 106 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: self-harm/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: self-harm/sd_baseline"
fi

# --- Job: violence / sd_baseline (est: 100 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: violence/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: violence/sd_baseline"
fi

# --- Job: hate / sd_baseline (est: 31 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: hate/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_hate.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: hate/sd_baseline"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 1 ALL DONE (7 jobs)"
