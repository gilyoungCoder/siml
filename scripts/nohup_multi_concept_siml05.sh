#!/usr/bin/env bash
# =============================================================================
# Multi-Concept Full Experiment Runner for siml-05 (8 GPUs)
# =============================================================================
# Generates per-GPU scripts and launches them all with nohup.
# Includes: 7 single concepts × 6 methods + compound concept experiments.
#
# Usage:
#   bash scripts/nohup_multi_concept_siml05.sh --dry-run     # preview only
#   bash scripts/nohup_multi_concept_siml05.sh --launch       # actually run
#   bash scripts/nohup_multi_concept_siml05.sh --status       # check progress
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAS_DIR="${REPO_ROOT}/CAS_SpatialCFG"
SAFREE_DIR="${REPO_ROOT}/SAFREE"
CONCEPT_PACKS="${REPO_ROOT}/docs/neurips_plan/multi_concept/concept_packs"
I2P_DIR="${SAFREE_DIR}/datasets/i2p_categories"
OUT_BASE="${CAS_DIR}/outputs/multi_concept"
LOG_DIR="${REPO_ROOT}/scripts/logs/multi_concept"

GEN_PYTHON="conda run -n sdd_copy --no-capture-output python"

NUM_GPUS=8

# =============================================================================
# Concept definitions
# =============================================================================
declare -A CONCEPT_CSV=(
    [sexual]="${I2P_DIR}/i2p_sexual.csv"
    [violence]="${I2P_DIR}/i2p_violence.csv"
    [harassment]="${I2P_DIR}/i2p_harassment.csv"
    [hate]="${I2P_DIR}/i2p_hate.csv"
    [shocking]="${I2P_DIR}/i2p_shocking.csv"
    [illegal_activity]="${I2P_DIR}/i2p_illegal_activity.csv"
    [self-harm]="${I2P_DIR}/i2p_self-harm.csv"
)

declare -A CONCEPT_CAS_THR=(
    [sexual]=0.6
    [violence]=0.5
    [harassment]=0.45
    [hate]=0.5
    [shocking]=0.5
    [illegal_activity]=0.5
    [self-harm]=0.45
)

declare -A CONCEPT_TARGET_WORDS=(
    [sexual]="nude naked nudity nsfw bare body"
    [violence]="blood wound weapon fight kill gore injury"
    [harassment]="bully threat harass intimidate mock"
    [hate]="hate racist slur discriminate"
    [shocking]="gore grotesque disturbing horror creepy"
    [illegal_activity]="drug steal crime illegal weapon"
    [self-harm]="suicide cut self-harm wound bleed"
)

declare -A CONCEPT_TARGET_CONCEPTS=(
    [sexual]="nudity nude_person naked_body"
    [violence]="violence blood weapon injury fight"
    [harassment]="harassment threat intimidation bullying"
    [hate]="hate discrimination racism slur"
    [shocking]="shocking gore grotesque disturbing"
    [illegal_activity]="illegal drugs crime theft weapon"
    [self-harm]="self-harm suicide cutting injury"
)

declare -A CONCEPT_ANCHOR_CONCEPTS=(
    [sexual]="clothed_person person_wearing_clothes"
    [violence]="peaceful_scene calm_landscape friendly_interaction"
    [harassment]="friendly_conversation cooperative_interaction"
    [hate]="diversity harmony cultural_respect equality"
    [shocking]="peaceful_scene beautiful_art calm_composition"
    [illegal_activity]="legal_activity professional_work normal_daily_life"
    [self-harm]="healing meditation healthy_activity wellness"
)

# Prompt counts for time estimation (approximate)
declare -A CONCEPT_PROMPTS=(
    [sexual]=931 [violence]=757 [harassment]=824
    [hate]=234 [shocking]=857 [illegal_activity]=727 [self-harm]=802
)

# =============================================================================
# Job generation: create all experiment commands
# =============================================================================
generate_jobs() {
    local jobs_file="$1"
    > "$jobs_file"

    local concepts="sexual violence harassment hate shocking illegal_activity self-harm"

    # --- Single concept experiments (42 jobs) ---
    for concept in $concepts; do
        local csv="${CONCEPT_CSV[$concept]}"
        local cas_thr="${CONCEPT_CAS_THR[$concept]}"
        local tw="${CONCEPT_TARGET_WORDS[$concept]}"
        local tc="${CONCEPT_TARGET_CONCEPTS[$concept]}"
        local ac="${CONCEPT_ANCHOR_CONCEPTS[$concept]}"
        local np="${CONCEPT_PROMPTS[$concept]}"
        local pack="${CONCEPT_PACKS}/${concept}"

        local concept_args=""
        if [[ -f "${pack}/metadata.json" ]]; then
            concept_args="--concept_packs ${pack}"
        else
            concept_args="--target_concepts ${tc} --anchor_concepts ${ac}"
        fi

        # 1. SD Baseline (~8 sec/prompt)
        local est_min=$(( np * 8 / 60 ))
        echo "${est_min}|${concept}|sd_baseline|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_baseline.py --prompts ${csv} --outdir ${OUT_BASE}/${concept}/sd_baseline --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

        # 2. SAFREE (~21 sec/prompt)
        est_min=$(( np * 21 / 60 ))
        echo "${est_min}|${concept}|safree|cd ${SAFREE_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_safree.py --data ${csv} --save-dir ${OUT_BASE}/${concept}/safree --model_id CompVis/stable-diffusion-v1-4 --num-samples 1 --config configs/sd_config.json --device cuda:0 --erase-id std --safree --self_validation_filter --latent_re_attention --sf_alpha 0.01" >> "$jobs_file"

        # 3-4. v14 × 2 guide modes (~21 sec/prompt each)
        for gm in dag_adaptive hybrid; do
            est_min=$(( np * 21 / 60 ))
            echo "${est_min}|${concept}|v14_${gm}|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v14.py --prompts ${csv} --outdir ${OUT_BASE}/${concept}/v14_${gm}_ss3.0_st0.3 --cas_threshold ${cas_thr} --guide_mode ${gm} --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words ${tw} ${concept_args} --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"
        done

        # 5-6. v19 × 2 guide modes (~21 sec/prompt each)
        for gm in dag_adaptive hybrid; do
            est_min=$(( np * 21 / 60 ))
            echo "${est_min}|${concept}|v19_${gm}|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v19.py --prompts ${csv} --outdir ${OUT_BASE}/${concept}/v19_text_${gm}_ss3.0_st0.3 --cas_threshold ${cas_thr} --guide_mode ${gm} --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words ${tw} ${concept_args} --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"
        done
    done

    # --- Compound concept experiments ---
    # Use sexual I2P (largest) as test prompts for compound erasing
    local compound_csv="${I2P_DIR}/i2p_sexual.csv"
    local est_min=$(( 931 * 21 / 60 ))

    # Compound 1: sexual + violence (2 concepts)
    echo "${est_min}|compound_sex_vio|v14_compound|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v14.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/sexual_violence/v14_dag_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked blood weapon fight --concept_packs ${CONCEPT_PACKS}/sexual ${CONCEPT_PACKS}/violence --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    echo "${est_min}|compound_sex_vio|v19_compound|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v19.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/sexual_violence/v19_dag_ss3.0_st0.3 --cas_threshold 0.5 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked blood weapon fight --concept_packs ${CONCEPT_PACKS}/sexual ${CONCEPT_PACKS}/violence --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    # Compound 2: sexual + violence + harassment (3 concepts)
    echo "${est_min}|compound_3way|v14_compound3|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v14.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/sex_vio_harass/v14_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words nude naked blood weapon fight bully threat --concept_packs ${CONCEPT_PACKS}/sexual ${CONCEPT_PACKS}/violence ${CONCEPT_PACKS}/harassment --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    echo "${est_min}|compound_3way|v19_compound3|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v19.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/sex_vio_harass/v19_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words nude naked blood weapon fight bully threat --concept_packs ${CONCEPT_PACKS}/sexual ${CONCEPT_PACKS}/violence ${CONCEPT_PACKS}/harassment --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    # Compound 3: ALL 7 concepts simultaneously
    local all_packs="${CONCEPT_PACKS}/sexual ${CONCEPT_PACKS}/violence ${CONCEPT_PACKS}/harassment ${CONCEPT_PACKS}/hate ${CONCEPT_PACKS}/shocking ${CONCEPT_PACKS}/illegal_activity ${CONCEPT_PACKS}/self-harm"
    local all_tw="nude naked blood weapon fight bully threat hate gore drug suicide"

    echo "${est_min}|compound_all7|v14_all7|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v14.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/all_7_concepts/v14_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode fused --probe_source text --exemplar_mode text --target_words ${all_tw} --concept_packs ${all_packs} --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    echo "${est_min}|compound_all7|v19_all7|cd ${CAS_DIR} && \${GPU_ENV} ${GEN_PYTHON} generate_v19.py --prompts ${compound_csv} --outdir ${OUT_BASE}/compound/all_7_concepts/v19_dag_ss3.0_st0.3 --cas_threshold 0.45 --guide_mode dag_adaptive --safety_scale 3.0 --spatial_threshold 0.3 --where_mode multi_probe --probe_source text --exemplar_selection all --target_words ${all_tw} --concept_packs ${all_packs} --nsamples 1 --steps 50 --cfg_scale 7.5 --seed 42" >> "$jobs_file"

    echo "Generated $(wc -l < "$jobs_file") jobs total"
}

# =============================================================================
# LPT scheduling: assign jobs to GPUs balancing total time
# =============================================================================
assign_jobs_to_gpus() {
    local jobs_file="$1"
    local assignments_file="$2"

    # Sort jobs by estimated time descending (LPT = Longest Processing Time first)
    sort -t'|' -k1 -nr "$jobs_file" > "${jobs_file}.sorted"

    # Track total time per GPU
    declare -a gpu_load
    for ((g=0; g<NUM_GPUS; g++)); do
        gpu_load[$g]=0
    done

    > "$assignments_file"

    while IFS='|' read -r est_min concept method cmd; do
        # Find GPU with least total load
        local min_gpu=0
        local min_load=${gpu_load[0]}
        for ((g=1; g<NUM_GPUS; g++)); do
            if (( gpu_load[g] < min_load )); then
                min_gpu=$g
                min_load=${gpu_load[$g]}
            fi
        done

        gpu_load[$min_gpu]=$(( gpu_load[$min_gpu] + est_min ))
        echo "${min_gpu}|${est_min}|${concept}|${method}|${cmd}" >> "$assignments_file"
    done < "${jobs_file}.sorted"

    echo ""
    echo "GPU Load Distribution (estimated minutes):"
    for ((g=0; g<NUM_GPUS; g++)); do
        local hours=$(( gpu_load[g] / 60 ))
        local mins=$(( gpu_load[g] % 60 ))
        echo "  GPU ${g}: ${gpu_load[g]} min (~${hours}h ${mins}m)"
    done
    echo ""
}

# =============================================================================
# Generate per-GPU runner scripts
# =============================================================================
generate_gpu_scripts() {
    local assignments_file="$1"

    mkdir -p "$LOG_DIR"

    for ((g=0; g<NUM_GPUS; g++)); do
        local script="${LOG_DIR}/gpu_${g}.sh"
        cat > "$script" <<HEADER
#!/usr/bin/env bash
# Auto-generated GPU ${g} runner for multi-concept experiments
set -euo pipefail
export GPU_ENV="CUDA_VISIBLE_DEVICES=${g}"

echo "[\$(date '+%Y-%m-%d %H:%M:%S')] GPU ${g} starting"

HEADER

        local job_count=0
        while IFS='|' read -r gpu est_min concept method cmd; do
            [[ "$gpu" -ne "$g" ]] && continue
            job_count=$((job_count + 1))

            # Extract outdir from command for skip check
            local outdir
            outdir=$(echo "$cmd" | grep -oP '(?<=--outdir )\S+' || echo "")

            cat >> "$script" <<JOB
# --- Job: ${concept} / ${method} (est: ${est_min} min) ---
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${concept}/${method}"
if [[ -n "${outdir}" && -f "${outdir}/stats.json" ]]; then
    echo "  [SKIP] Already completed"
else
    ${cmd} 2>&1 | tail -5
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Finished: ${concept}/${method}"
fi

JOB
        done < "$assignments_file"

        cat >> "$script" <<FOOTER
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] GPU ${g} ALL DONE (${job_count} jobs)"
FOOTER

        chmod +x "$script"
        echo "  Created: ${script} (${job_count} jobs)"
    done
}

# =============================================================================
# Launch with nohup
# =============================================================================
launch_all() {
    echo "Launching 8 GPU runners with nohup..."
    mkdir -p "$LOG_DIR"

    for ((g=0; g<NUM_GPUS; g++)); do
        local script="${LOG_DIR}/gpu_${g}.sh"
        local log="${LOG_DIR}/gpu_${g}.log"
        if [[ -f "$script" ]]; then
            nohup bash "$script" > "$log" 2>&1 &
            echo "  GPU ${g}: PID $! -> ${log}"
        fi
    done

    echo ""
    echo "All launched! Monitor with:"
    echo "  bash $0 --status"
    echo "  tail -f ${LOG_DIR}/gpu_*.log"
}

# =============================================================================
# Status check
# =============================================================================
check_status() {
    echo "=== Multi-Concept Experiment Status ==="
    echo ""

    # Running processes
    echo "Running GPU processes:"
    for ((g=0; g<NUM_GPUS; g++)); do
        local log="${LOG_DIR}/gpu_${g}.log"
        if [[ -f "$log" ]]; then
            local last_line
            last_line=$(tail -1 "$log" 2>/dev/null || echo "no log")
            local running="?"
            if echo "$last_line" | grep -q "ALL DONE"; then
                running="DONE"
            else
                running="RUNNING"
            fi
            echo "  GPU ${g}: ${running} | ${last_line}"
        fi
    done

    echo ""
    echo "Completed experiments:"
    local total=0 done_count=0
    for concept_dir in "${OUT_BASE}"/*/; do
        [[ -d "$concept_dir" ]] || continue
        local concept
        concept=$(basename "$concept_dir")
        for method_dir in "${concept_dir}"/*/; do
            [[ -d "$method_dir" ]] || continue
            total=$((total + 1))
            local method
            method=$(basename "$method_dir")
            local imgs
            imgs=$(find "$method_dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
            if [[ -f "${method_dir}/stats.json" ]]; then
                done_count=$((done_count + 1))
                echo "  [DONE] ${concept}/${method} (${imgs} images)"
            elif [[ "$imgs" -gt 0 ]]; then
                echo "  [WIP]  ${concept}/${method} (${imgs} images)"
            fi
        done
    done
    echo ""
    echo "Progress: ${done_count}/${total} experiments completed"
}

# =============================================================================
# Main
# =============================================================================
ACTION="${1:---help}"

case "$ACTION" in
    --dry-run)
        echo "=== DRY RUN: Multi-Concept Experiment Plan ==="
        echo ""
        JOBS_FILE=$(mktemp)
        ASSIGN_FILE=$(mktemp)
        generate_jobs "$JOBS_FILE"
        assign_jobs_to_gpus "$JOBS_FILE" "$ASSIGN_FILE"

        echo "Per-GPU job assignments:"
        for ((g=0; g<NUM_GPUS; g++)); do
            echo ""
            echo "--- GPU ${g} ---"
            while IFS='|' read -r gpu est_min concept method cmd; do
                [[ "$gpu" -ne "$g" ]] && continue
                echo "  [${est_min}min] ${concept} / ${method}"
            done < "$ASSIGN_FILE"
        done
        rm -f "$JOBS_FILE" "${JOBS_FILE}.sorted" "$ASSIGN_FILE"
        ;;

    --generate)
        echo "=== Generating GPU scripts ==="
        JOBS_FILE=$(mktemp)
        ASSIGN_FILE=$(mktemp)
        generate_jobs "$JOBS_FILE"
        assign_jobs_to_gpus "$JOBS_FILE" "$ASSIGN_FILE"
        generate_gpu_scripts "$ASSIGN_FILE"
        echo ""
        echo "Scripts ready at: ${LOG_DIR}/gpu_*.sh"
        echo "Review them, then run: bash $0 --launch"
        rm -f "$JOBS_FILE" "${JOBS_FILE}.sorted" "$ASSIGN_FILE"
        ;;

    --launch)
        if [[ ! -f "${LOG_DIR}/gpu_0.sh" ]]; then
            echo "GPU scripts not found. Run --generate first."
            exit 1
        fi
        launch_all
        ;;

    --status)
        check_status
        ;;

    --help|*)
        cat <<EOF
Multi-Concept Full Experiment Runner (siml-05, 8 GPUs)

Steps:
  1. bash $0 --dry-run      # Preview job distribution
  2. bash $0 --generate     # Create per-GPU scripts
  3. bash $0 --launch       # Launch all with nohup

Monitoring:
  bash $0 --status          # Check progress

Total: 42 single-concept + 6 compound = 48 experiments
Estimated wall time: ~20-21 hours (8 GPUs parallel)
EOF
        ;;
esac
