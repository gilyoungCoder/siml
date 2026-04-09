#!/usr/bin/env bash
# Auto-generated GPU 4 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=4"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 4 starting"

# --- Job: sexual / safree (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: sexual/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/sexual/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: sexual/safree"
fi

# --- Job: shocking / v19_dag_adaptive (est: 299 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: shocking/v19_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v19_text_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v19_text_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_shocking.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v19_text_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words gore grotesque disturbing horror creepy --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/shocking --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: shocking/v19_dag_adaptive"
fi

# --- Job: harassment / v19_dag_adaptive (est: 288 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: harassment/v19_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v19_text_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v19_text_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_harassment.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v19_text_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words bully threat harass intimidate mock --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: harassment/v19_dag_adaptive"
fi

# --- Job: self-harm / safree (est: 280 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: self-harm/safree"
if [[ -n "" && -f "/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/SAFREE && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_safree.py --data /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_self-harm.csv --save-dir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/self-harm/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 4 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: self-harm/safree"
fi

# --- Job: shocking / sd_baseline (est: 114 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: shocking/sd_baseline"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/sd_baseline" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/sd_baseline/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_baseline.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_shocking.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/sd_baseline --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: shocking/sd_baseline"
fi

# --- Job: hate / v19_hybrid (est: 81 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: hate/v19_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v19_text_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v19_text_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_hate.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v19_text_hybrid_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words hate racist slur discriminate --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/hate --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: hate/v19_hybrid"
fi

# --- Job: hate / v14_hybrid (est: 81 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: hate/v14_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v14_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v14_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_hate.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/hate/v14_hybrid_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words hate racist slur discriminate --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/hate --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: hate/v14_hybrid"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 4 ALL DONE (7 jobs)"
