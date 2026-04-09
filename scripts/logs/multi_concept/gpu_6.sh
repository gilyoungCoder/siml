#!/usr/bin/env bash
# Auto-generated GPU 6 runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=6"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 6 starting"

# --- Job: compound_sex_vio / v14_compound (est: 325 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: compound_sex_vio/v14_compound"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sexual_violence/v14_dag_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sexual_violence/v14_dag_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_sexual.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/compound/sexual_violence/v14_dag_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked blood weapon fight --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/sexual /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: compound_sex_vio/v14_compound"
fi

# --- Job: shocking / v14_dag_adaptive (est: 299 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: shocking/v14_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v14_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v14_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_shocking.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/shocking/v14_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words gore grotesque disturbing horror creepy --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/shocking --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: shocking/v14_dag_adaptive"
fi

# --- Job: harassment / v14_dag_adaptive (est: 288 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: harassment/v14_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v14_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v14_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_harassment.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/harassment/v14_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words bully threat harass intimidate mock --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/harassment --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: harassment/v14_dag_adaptive"
fi

# --- Job: violence / v19_dag_adaptive (est: 264 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: violence/v19_dag_adaptive"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v19_text_dag_adaptive_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v19_text_dag_adaptive_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v19.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_violence.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/violence/v19_text_dag_adaptive_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words blood wound weapon fight kill gore injury --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/violence --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: violence/v19_dag_adaptive"
fi

# --- Job: illegal_activity / v14_hybrid (est: 254 min) ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: illegal_activity/v14_hybrid"
if [[ -n "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_hybrid_ss3.0_st0.3" && -f "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_hybrid_ss3.0_st0.3/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    cd /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG && ${GPU_ENV} conda run -n sdd_copy --no-capture-output python generate_v14.py --prompts /mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv --outdir /mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/multi_concept/illegal_activity/v14_hybrid_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode hybrid --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words drug steal crime illegal weapon --concept_packs /mnt/home3/yhgil99/unlearning/docs/neurips_plan/multi_concept/concept_packs/illegal_activity --nsamples 4 --steps 50 --cfg_scale 7.5 --seed 42 2>&1 | tail -5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: illegal_activity/v14_hybrid"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 6 ALL DONE (5 jobs)"
