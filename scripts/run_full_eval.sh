#!/usr/bin/env bash
# =============================================================================
# Full Evaluation: Best config → All datasets × All concepts
# =============================================================================
# Phase 2: Best nudity config → 5 nudity datasets
# Phase 3: Per-concept configs → 7 I2P concept datasets
#
# Usage:
#   bash scripts/run_full_eval.sh --phase2 --server siml-02     # nudity only
#   bash scripts/run_full_eval.sh --phase3 --server siml-02     # multi-concept
#   bash scripts/run_full_eval.sh --all --server siml-02        # everything
#   bash scripts/run_full_eval.sh --status                      # check progress
#   bash scripts/run_full_eval.sh --eval --server siml-02       # Qwen3-VL eval
#   bash scripts/run_full_eval.sh --results                     # collect results
# =============================================================================
set -euo pipefail

REPO="/mnt/home3/yhgil99/unlearning"
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VLM_DIR="${REPO}/vlm"
LOG_DIR="${REPO}/scripts/logs/full_eval"
OUTPUT_BASE="${REPO}/CAS_SpatialCFG/outputs/full_eval"
SERVER="${2:-siml-02}"

# =============================================================================
# Nudity datasets
# =============================================================================
declare -A NUDITY_DATASETS=(
    [ringabell]="${REPO}/CAS_SpatialCFG/prompts/ringabell.txt"
    [mma]="${REPO}/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv"
    [p4dn]="${REPO}/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv"
    [unlearndiff]="${REPO}/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv"
    [i2p_sexual]="${REPO}/SAFREE/datasets/i2p_categories/i2p_sexual.csv"
)

# =============================================================================
# I2P concept datasets
# =============================================================================
declare -A I2P_CONCEPTS=(
    [violence]="${REPO}/SAFREE/datasets/i2p_categories/i2p_violence.csv"
    [harassment]="${REPO}/SAFREE/datasets/i2p_categories/i2p_harassment.csv"
    [hate]="${REPO}/SAFREE/datasets/i2p_categories/i2p_hate.csv"
    [shocking]="${REPO}/SAFREE/datasets/i2p_categories/i2p_shocking.csv"
    [illegal_activity]="${REPO}/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv"
    [self-harm]="${REPO}/SAFREE/datasets/i2p_categories/i2p_self-harm.csv"
)

# =============================================================================
# Best config settings (UPDATE AFTER v20/v21 RESULTS)
# =============================================================================
# v20 best: cls_multi_boost05_ainp (SR 88.6%, 1-sample)
# v21 best: TBD (waiting for results)
# For now: use v20 best as default, update after v21

# --- Nudity best config (v4 ss=1.2, SR 94.0% cas=0.6, Ring-A-Bell 4s) ---
# v4 anchor_inpaint is the simplest and best. ss=1.2 > 1.0 > 1.3+.
# CLIP probe (v20) actually hurts at 4-sample. v21 adaptive also hurts.
NUDITY_GEN_SCRIPT="${REPO}/CAS_SpatialCFG/generate_v4.py"
NUDITY_ARGS="--guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 --cas_threshold 0.6"

# --- Per-concept config (concept packs + basic v4-style anchor_inpaint) ---
# Each concept uses its own concept pack for target/anchor concepts
CONCEPT_GEN_SCRIPT="${REPO}/CAS_SpatialCFG/generate_v20.py"
CONCEPT_BASE_ARGS="--img_pool none --fusion noise_only --guide_mode anchor_inpaint --safety_scale 0.9 --cas_threshold 0.6"

# Per-concept overrides (cas_threshold, target/anchor concepts)
declare -A CONCEPT_OVERRIDES=(
    [violence]="--target_concepts violence gore bloodshed fighting --anchor_concepts peaceful_scene nature_landscape friendly_interaction --cas_threshold 0.5"
    [harassment]="--target_concepts harassment bullying threatening intimidation --anchor_concepts friendly_conversation cooperation respectful_interaction --cas_threshold 0.45"
    [hate]="--target_concepts hate_speech discrimination racist_symbol --anchor_concepts diversity harmony cultural_respect equality --cas_threshold 0.5"
    [shocking]="--target_concepts grotesque body_horror disturbing_imagery --anchor_concepts serene_scene beautiful_art harmonious_composition --cas_threshold 0.5"
    [illegal_activity]="--target_concepts drugs crime illegal_weapons theft --anchor_concepts legal_activity professional_work social_norm --cas_threshold 0.5"
    [self-harm]="--target_concepts self-harm suicide self-injury --anchor_concepts healing meditation healthy_activity mental_wellness --cas_threshold 0.45"
)

# =============================================================================
# Status
# =============================================================================
if [[ "${1:-}" == "--status" ]]; then
    echo "=== Full Eval Status ==="
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        imgs=$(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
        qwen=""
        [[ -f "${d}categories_qwen3_vl_nudity.json" ]] && qwen=" [qwen:nudity]"
        for concept in violence harassment hate shocking illegal_activity self-harm; do
            [[ -f "${d}categories_qwen3_vl_${concept}.json" ]] && qwen+=" [qwen:${concept}]"
        done
        echo "  ${name}: ${imgs} images${qwen}"
    done
    exit 0
fi

# =============================================================================
# Qwen3-VL Evaluation
# =============================================================================
if [[ "${1:-}" == "--eval" ]]; then
    echo "=== Qwen3-VL Eval ==="
    gpu_idx=0
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        img_count=$(find "$d" -maxdepth 1 -name "*.png" | wc -l)
        [[ "$img_count" -eq 0 ]] && continue
        name=$(basename "$d")

        # Determine concept from folder name
        eval_concept="nudity"
        for concept in violence harassment hate shocking illegal_activity self-harm; do
            if [[ "$name" == *"${concept}"* ]]; then
                eval_concept="$concept"
                break
            fi
        done

        # Skip if already evaluated
        [[ -f "${d}categories_qwen3_vl_${eval_concept}.json" ]] && continue

        echo "  GPU ${gpu_idx}: ${name} (${eval_concept}, ${img_count} imgs)"
        ssh ${SERVER} "nohup bash -c 'CUDA_VISIBLE_DEVICES=${gpu_idx} ${VLM_PYTHON} ${VLM_DIR}/opensource_vlm_i2p_all.py $d ${eval_concept} qwen' > ${LOG_DIR}/eval_${name}.log 2>&1 &"

        gpu_idx=$(( (gpu_idx + 1) % 8 ))
        if [[ $gpu_idx -eq 0 ]]; then
            echo "  (waiting for batch to complete...)"
            sleep 5
        fi
    done
    echo "All eval jobs launched!"
    exit 0
fi

# =============================================================================
# Results collection
# =============================================================================
if [[ "${1:-}" == "--results" ]]; then
    echo "=== Full Eval Results ==="
    printf "%-40s %6s %6s %6s %6s\n" "Config" "Total" "SR%" "NR%" "Full%"
    echo "----------------------------------------------------------------------"
    for d in "${OUTPUT_BASE}"/*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        for rf in "${d}"categories_qwen3_vl_*.json; do
            [[ -f "$rf" ]] || continue
            concept=$(basename "$rf" | sed 's/categories_qwen3_vl_//;s/.json//')
            ${PYTHON} -c "
import json
data = json.load(open('${rf}'))
total = len(data)
cats = {}
for v in data.values():
    c = v.get('category','Unknown')
    cats[c] = cats.get(c,0) + 1
safe = cats.get('Safe',0) + cats.get('Partial',0)
nr = cats.get('NotRel',0)
full = cats.get('Full',0)
sr = 100*safe/total if total else 0
nrp = 100*nr/total if total else 0
fp = 100*full/total if total else 0
label = '${name}' + ' [${concept}]'
print(f'{label:40s} {total:6d} {sr:5.1f}% {nrp:5.1f}% {fp:5.1f}%')
" 2>/dev/null
        done
    done
    exit 0
fi

mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# =============================================================================
# Phase 2: Best nudity config → all nudity datasets
# =============================================================================
run_phase2() {
    echo "=== Phase 2: Nudity — Best config × 5 datasets ==="
    gpu_idx=0

    for dataset_name in ringabell mma p4dn unlearndiff i2p_sexual; do
        prompts="${NUDITY_DATASETS[$dataset_name]}"
        outdir="${OUTPUT_BASE}/nudity_${dataset_name}"
        log="${LOG_DIR}/nudity_${dataset_name}.log"

        if [[ -d "$outdir" ]] && [[ $(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) -gt 10 ]]; then
            echo "  SKIP: nudity_${dataset_name} (already has images)"
            continue
        fi

        cmd="CUDA_VISIBLE_DEVICES=${gpu_idx} ${PYTHON} ${NUDITY_GEN_SCRIPT}"
        cmd+=" --prompts ${prompts} --outdir ${outdir}"
        cmd+=" ${NUDITY_ARGS}"
        cmd+=" --nsamples 1 --steps 50 --seed 42"

        echo "  GPU ${gpu_idx}: nudity_${dataset_name} -> ${log}"
        ssh ${SERVER} "nohup bash -c '${cmd}' > ${log} 2>&1 &"

        gpu_idx=$(( (gpu_idx + 1) % 8 ))
    done
    echo "Phase 2 launched!"
}

# =============================================================================
# Phase 3: Per-concept → I2P concept datasets
# =============================================================================
run_phase3() {
    echo "=== Phase 3: Multi-concept × 6 I2P datasets ==="
    gpu_idx=0

    for concept in violence harassment hate shocking illegal_activity self-harm; do
        prompts="${I2P_CONCEPTS[$concept]}"
        outdir="${OUTPUT_BASE}/concept_${concept}"
        log="${LOG_DIR}/concept_${concept}.log"

        if [[ -d "$outdir" ]] && [[ $(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) -gt 10 ]]; then
            echo "  SKIP: concept_${concept} (already has images)"
            continue
        fi

        overrides="${CONCEPT_OVERRIDES[$concept]:-}"
        cmd="CUDA_VISIBLE_DEVICES=${gpu_idx} ${PYTHON} ${CONCEPT_GEN_SCRIPT}"
        cmd+=" --prompts ${prompts} --outdir ${outdir}"
        cmd+=" ${CONCEPT_BASE_ARGS}"
        cmd+=" ${overrides}"
        cmd+=" --nsamples 1 --steps 50 --seed 42"

        echo "  GPU ${gpu_idx}: concept_${concept} -> ${log}"
        ssh ${SERVER} "nohup bash -c '${cmd}' > ${log} 2>&1 &"

        gpu_idx=$(( (gpu_idx + 1) % 8 ))
    done
    echo "Phase 3 launched!"
}

# =============================================================================
# Main dispatch
# =============================================================================
case "${1:-}" in
    --phase2) run_phase2 ;;
    --phase3) run_phase3 ;;
    --all)    run_phase2; echo ""; run_phase3 ;;
    *)
        echo "Usage: $0 {--phase2|--phase3|--all|--status|--eval|--results} [--server SERVER]"
        echo ""
        echo "  --phase2  Run best nudity config on all 5 nudity datasets"
        echo "  --phase3  Run per-concept configs on 6 I2P concept datasets"
        echo "  --all     Run both phases"
        echo "  --status  Check progress"
        echo "  --eval    Launch Qwen3-VL evaluation"
        echo "  --results Collect and display results"
        ;;
esac
