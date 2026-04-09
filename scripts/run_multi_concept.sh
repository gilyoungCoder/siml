#!/usr/bin/env bash
# =============================================================================
# Multi-Concept Experiment Pipeline
# =============================================================================
# Runs SD baseline, SAFREE, v14, v19 across all I2P concept categories.
#
# Usage:
#   bash scripts/run_multi_concept.sh --dry-run
#   bash scripts/run_multi_concept.sh --concepts "violence harassment" --gpus 1,2,3 --phase generate
#   bash scripts/run_multi_concept.sh --all --gpus 0,1 --phase generate --nohup
#   bash scripts/run_multi_concept.sh --phase eval --gpus 0
#   bash scripts/run_multi_concept.sh --phase aggregate
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAS_DIR="${REPO_ROOT}/CAS_SpatialCFG"
SAFREE_DIR="${REPO_ROOT}/SAFREE"
VLM_DIR="${REPO_ROOT}/vlm"
CONCEPT_PACKS_DIR="${REPO_ROOT}/docs/neurips_plan/multi_concept/concept_packs"
I2P_DIR="${SAFREE_DIR}/datasets/i2p_categories"
OUTPUT_BASE="${CAS_DIR}/outputs/multi_concept"

# Conda envs
GEN_PYTHON="conda run -n sdd_copy --no-capture-output python"
VLM_PYTHON="conda run -n vlm --no-capture-output python"

# All concepts and their I2P CSV files
ALL_CONCEPTS="sexual violence harassment hate shocking illegal_activity self-harm"

# Default hyperparameters for grid search (small initial grid)
SAFETY_SCALES="3.0"
SPATIAL_THRESHOLDS="0.3"
GUIDE_MODES="dag_adaptive hybrid"

# ============================================================================
# Concept config: read CAS threshold from metadata.json
# ============================================================================
get_cas_threshold() {
    local concept="$1"
    local meta="${CONCEPT_PACKS_DIR}/${concept}/metadata.json"
    if [[ -f "$meta" ]]; then
        python3 -c "import json; print(json.load(open('${meta}'))['cas_threshold'])"
    else
        echo "0.5"  # default
    fi
}

get_i2p_csv() {
    local concept="$1"
    local csv_file="${I2P_DIR}/i2p_${concept}.csv"
    if [[ -f "$csv_file" ]]; then
        echo "$csv_file"
    else
        echo ""
    fi
}

get_target_concepts() {
    local concept="$1"
    case "$concept" in
        sexual)           echo "nudity nude_person naked_body" ;;
        violence)         echo "violence blood weapon injury fight" ;;
        harassment)       echo "harassment threat intimidation bullying" ;;
        hate)             echo "hate discrimination racism slur" ;;
        shocking)         echo "shocking gore grotesque disturbing" ;;
        illegal_activity) echo "illegal drugs crime theft weapon" ;;
        self-harm)        echo "self-harm suicide cutting injury" ;;
        *)                echo "$concept" ;;
    esac
}

get_anchor_concepts() {
    local concept="$1"
    case "$concept" in
        sexual)           echo "clothed_person person_wearing_clothes" ;;
        violence)         echo "peaceful_scene calm_landscape friendly_interaction" ;;
        harassment)       echo "friendly_conversation cooperative_interaction" ;;
        hate)             echo "diversity harmony cultural_respect equality" ;;
        shocking)         echo "peaceful_scene beautiful_art calm_composition" ;;
        illegal_activity) echo "legal_activity professional_work normal_daily_life" ;;
        self-harm)        echo "healing meditation healthy_activity wellness" ;;
        *)                echo "safe normal" ;;
    esac
}

get_target_words() {
    local concept="$1"
    case "$concept" in
        sexual)           echo "nude naked nudity nsfw bare body" ;;
        violence)         echo "blood wound weapon fight kill gore injury" ;;
        harassment)       echo "bully threat harass intimidate mock" ;;
        hate)             echo "hate racist slur discriminate" ;;
        shocking)         echo "gore grotesque disturbing horror creepy" ;;
        illegal_activity) echo "drug steal crime illegal weapon" ;;
        self-harm)        echo "suicide cut self-harm wound bleed" ;;
        *)                echo "$concept" ;;
    esac
}

# ============================================================================
# Generation functions
# ============================================================================
run_sd_baseline() {
    local concept="$1" gpu="$2" csv_file="$3"
    local outdir="${OUTPUT_BASE}/${concept}/sd_baseline"

    if [[ -f "${outdir}/stats.json" ]]; then
        echo "  [SKIP] sd_baseline for ${concept} (already done)"
        return 0
    fi

    echo "  [RUN] sd_baseline for ${concept} on GPU ${gpu}"
    local cmd="cd ${CAS_DIR} && CUDA_VISIBLE_DEVICES=${gpu} ${GEN_PYTHON} generate_baseline.py \
        --prompts ${csv_file} \
        --outdir ${outdir} \
        --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    CMD: $cmd"
    else
        eval "$cmd"
    fi
}

run_safree() {
    local concept="$1" gpu="$2" csv_file="$3"
    local outdir="${OUTPUT_BASE}/${concept}/safree"

    if [[ -f "${outdir}/stats.json" ]] || [[ -d "${outdir}" && $(ls "${outdir}"/*.png 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "  [SKIP] safree for ${concept} (already done)"
        return 0
    fi

    echo "  [RUN] safree for ${concept} on GPU ${gpu}"
    local cmd="cd ${SAFREE_DIR} && CUDA_VISIBLE_DEVICES=${gpu} ${GEN_PYTHON} generate_safree.py \
        --data ${csv_file} \
        --save-dir ${outdir} \
        --model_id CompVis/stable-diffusion-v1-4 \
        --num-samples 1 \
        --config configs/sd_config.json \
        --device cuda:0 \
        --erase-id std \
        --safree \
        --self_validation_filter \
        --latent_re_attention \
        --sf_alpha 0.01"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    CMD: $cmd"
    else
        eval "$cmd"
    fi
}

run_v14() {
    local concept="$1" gpu="$2" csv_file="$3"
    local cas_thr
    cas_thr=$(get_cas_threshold "$concept")
    local target_concepts
    target_concepts=$(get_target_concepts "$concept")
    local anchor_concepts
    anchor_concepts=$(get_anchor_concepts "$concept")
    local target_words
    target_words=$(get_target_words "$concept")
    local pack_dir="${CONCEPT_PACKS_DIR}/${concept}"

    for guide_mode in $GUIDE_MODES; do
        for ss in $SAFETY_SCALES; do
            for st in $SPATIAL_THRESHOLDS; do
                local config_name="v14_${guide_mode}_ss${ss}_st${st}"
                local outdir="${OUTPUT_BASE}/${concept}/${config_name}"

                if [[ -f "${outdir}/stats.json" ]]; then
                    echo "  [SKIP] ${config_name} for ${concept}"
                    continue
                fi

                echo "  [RUN] ${config_name} for ${concept} on GPU ${gpu}"

                # Use concept_packs if available, else fallback to manual concepts
                local concept_args=""
                if [[ -f "${pack_dir}/metadata.json" ]]; then
                    concept_args="--concept_packs ${pack_dir}"
                else
                    concept_args="--target_concepts ${target_concepts} --anchor_concepts ${anchor_concepts}"
                fi

                local cmd="cd ${CAS_DIR} && CUDA_VISIBLE_DEVICES=${gpu} ${GEN_PYTHON} generate_v14.py \
                    --prompts ${csv_file} \
                    --outdir ${outdir} \
                    --cas_threshold ${cas_thr} \
                    --guide_mode ${guide_mode} \
                    --safety_scale ${ss} \
                    --spatial_threshold ${st} \
                    --where_mode fused \
                    --probe_source text \
                    --exemplar_mode text \
                    --target_words ${target_words} \
                    ${concept_args} \
                    --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "    CMD: $cmd"
                else
                    eval "$cmd"
                fi
            done
        done
    done
}

run_v19() {
    local concept="$1" gpu="$2" csv_file="$3"
    local cas_thr
    cas_thr=$(get_cas_threshold "$concept")
    local target_concepts
    target_concepts=$(get_target_concepts "$concept")
    local anchor_concepts
    anchor_concepts=$(get_anchor_concepts "$concept")
    local target_words
    target_words=$(get_target_words "$concept")
    local pack_dir="${CONCEPT_PACKS_DIR}/${concept}"

    # Check for CLIP embeddings (if available, use both; else text only)
    local clip_emb="${pack_dir}/clip_exemplar_embeddings.pt"
    local probe_source="text"
    local clip_args=""
    if [[ -f "$clip_emb" ]]; then
        probe_source="both"
        clip_args="--clip_embeddings ${clip_emb}"
    fi

    for guide_mode in $GUIDE_MODES; do
        for ss in $SAFETY_SCALES; do
            for st in $SPATIAL_THRESHOLDS; do
                local config_name="v19_${probe_source}_${guide_mode}_ss${ss}_st${st}"
                local outdir="${OUTPUT_BASE}/${concept}/${config_name}"

                if [[ -f "${outdir}/stats.json" ]]; then
                    echo "  [SKIP] ${config_name} for ${concept}"
                    continue
                fi

                echo "  [RUN] ${config_name} for ${concept} on GPU ${gpu}"

                local concept_args=""
                if [[ -f "${pack_dir}/metadata.json" ]]; then
                    concept_args="--concept_packs ${pack_dir}"
                else
                    concept_args="--target_concepts ${target_concepts} --anchor_concepts ${anchor_concepts}"
                fi

                local cmd="cd ${CAS_DIR} && CUDA_VISIBLE_DEVICES=${gpu} ${GEN_PYTHON} generate_v19.py \
                    --prompts ${csv_file} \
                    --outdir ${outdir} \
                    --cas_threshold ${cas_thr} \
                    --guide_mode ${guide_mode} \
                    --safety_scale ${ss} \
                    --spatial_threshold ${st} \
                    --where_mode multi_probe \
                    --probe_source ${probe_source} \
                    --exemplar_selection all \
                    --target_words ${target_words} \
                    ${clip_args} \
                    ${concept_args} \
                    --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42"

                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "    CMD: $cmd"
                else
                    eval "$cmd"
                fi
            done
        done
    done
}

# ============================================================================
# Evaluation function
# ============================================================================
run_eval() {
    local concept="$1" gpu="$2"
    local eval_concept="$concept"
    # Map concept names to VLM eval names
    case "$concept" in
        sexual)           eval_concept="nudity" ;;
        illegal_activity) eval_concept="illegal" ;;
        self-harm)        eval_concept="self_harm" ;;
    esac

    local concept_dir="${OUTPUT_BASE}/${concept}"
    if [[ ! -d "$concept_dir" ]]; then
        echo "  [SKIP] No outputs for ${concept}"
        return 0
    fi

    for method_dir in "${concept_dir}"/*/; do
        local method_name
        method_name=$(basename "$method_dir")

        # Check if already evaluated
        local has_result=false
        for fname in "categories_qwen_${eval_concept}.json" \
                     "categories_qwen2_vl_${eval_concept}.json" \
                     "categories_qwen3_vl_${eval_concept}.json"; do
            if [[ -f "${method_dir}${fname}" ]]; then
                has_result=true
                break
            fi
        done

        if [[ "$has_result" == "true" ]]; then
            echo "  [SKIP] eval ${concept}/${method_name} (already done)"
            continue
        fi

        # Check for images
        local img_count
        img_count=$(ls "${method_dir}"*.png 2>/dev/null | wc -l)
        if [[ "$img_count" -eq 0 ]]; then
            echo "  [SKIP] eval ${concept}/${method_name} (no images)"
            continue
        fi

        echo "  [EVAL] ${concept}/${method_name} (${img_count} images) on GPU ${gpu}"
        local cmd="cd ${VLM_DIR} && CUDA_VISIBLE_DEVICES=${gpu} ${VLM_PYTHON} opensource_vlm_i2p_all.py \
            ${method_dir} ${eval_concept} qwen"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "    CMD: $cmd"
        else
            eval "$cmd"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================
usage() {
    cat <<EOF
Multi-Concept Experiment Pipeline

Usage:
  $0 [OPTIONS]

Options:
  --concepts "c1 c2 ..."   Concepts to run (default: all 7)
  --methods "m1 m2 ..."    Methods to run (default: "sd_baseline safree v14 v19")
  --gpus 0,1,2             Comma-separated GPU IDs to use
  --phase PHASE             Phase to run: generate|eval|aggregate|all (default: all)
  --all                    Run all concepts
  --dry-run                Print commands without executing
  --nohup                  Run generation in background with nohup
  --safety-scales "s1 s2"  Override safety_scale grid (default: "$SAFETY_SCALES")
  --spatial-thresholds "t1 t2"  Override spatial_threshold grid (default: "$SPATIAL_THRESHOLDS")
  --guide-modes "m1 m2"    Override guide_mode grid (default: "$GUIDE_MODES")
  -h, --help               Show this help

Concepts: $ALL_CONCEPTS
Methods:  sd_baseline safree v14 v19
EOF
    exit 0
}

DRY_RUN="false"
USE_NOHUP="false"
PHASE="all"
CONCEPTS=""
METHODS="sd_baseline safree v14 v19"
GPU_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --concepts)       CONCEPTS="$2"; shift 2 ;;
        --methods)        METHODS="$2"; shift 2 ;;
        --gpus)           GPU_LIST="$2"; shift 2 ;;
        --phase)          PHASE="$2"; shift 2 ;;
        --all)            CONCEPTS="$ALL_CONCEPTS"; shift ;;
        --dry-run)        DRY_RUN="true"; shift ;;
        --nohup)          USE_NOHUP="true"; shift ;;
        --safety-scales)  SAFETY_SCALES="$2"; shift 2 ;;
        --spatial-thresholds) SPATIAL_THRESHOLDS="$2"; shift 2 ;;
        --guide-modes)    GUIDE_MODES="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

# Default: all concepts
if [[ -z "$CONCEPTS" ]]; then
    CONCEPTS="$ALL_CONCEPTS"
fi

# Default: require GPU specification
if [[ -z "$GPU_LIST" ]]; then
    echo "ERROR: --gpus is required. Check server sheet first!"
    echo "Example: --gpus 1,2,3"
    exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "=============================================="
echo "Multi-Concept Experiment Pipeline"
echo "=============================================="
echo "  Concepts: $CONCEPTS"
echo "  Methods:  $METHODS"
echo "  GPUs:     ${GPUS[*]} (${NUM_GPUS} GPUs)"
echo "  Phase:    $PHASE"
echo "  Dry-run:  $DRY_RUN"
echo "  Grid:     ss=${SAFETY_SCALES} st=${SPATIAL_THRESHOLDS} guide=${GUIDE_MODES}"
echo "  Output:   $OUTPUT_BASE"
echo "=============================================="

# ---- Phase: Generate ----
if [[ "$PHASE" == "generate" || "$PHASE" == "all" ]]; then
    echo ""
    echo ">>> Phase 1: Image Generation"
    echo ""

    gpu_idx=0
    for concept in $CONCEPTS; do
        csv_file=$(get_i2p_csv "$concept")
        if [[ -z "$csv_file" ]]; then
            echo "[WARN] No I2P CSV for concept: ${concept}, skipping"
            continue
        fi

        echo "--- Concept: ${concept} ($(wc -l < "$csv_file") lines) ---"
        local_gpu="${GPUS[$((gpu_idx % NUM_GPUS))]}"

        for method in $METHODS; do
            case "$method" in
                sd_baseline) run_sd_baseline "$concept" "$local_gpu" "$csv_file" ;;
                safree)      run_safree "$concept" "$local_gpu" "$csv_file" ;;
                v14)         run_v14 "$concept" "$local_gpu" "$csv_file" ;;
                v19)         run_v19 "$concept" "$local_gpu" "$csv_file" ;;
                *)           echo "[WARN] Unknown method: $method" ;;
            esac
        done

        gpu_idx=$((gpu_idx + 1))
    done
fi

# ---- Phase: Evaluate ----
if [[ "$PHASE" == "eval" || "$PHASE" == "all" ]]; then
    echo ""
    echo ">>> Phase 2: Qwen VLM Evaluation"
    echo ""

    gpu_idx=0
    for concept in $CONCEPTS; do
        local_gpu="${GPUS[$((gpu_idx % NUM_GPUS))]}"
        echo "--- Evaluating: ${concept} on GPU ${local_gpu} ---"
        run_eval "$concept" "$local_gpu"
        gpu_idx=$((gpu_idx + 1))
    done
fi

# ---- Phase: Aggregate ----
if [[ "$PHASE" == "aggregate" || "$PHASE" == "all" ]]; then
    echo ""
    echo ">>> Phase 3: Result Aggregation"
    echo ""
    python3 "${REPO_ROOT}/scripts/aggregate_multi_concept_results.py" \
        --base-dir "$OUTPUT_BASE" \
        --csv "${OUTPUT_BASE}/results_summary.csv"
fi

echo ""
echo "Done!"
