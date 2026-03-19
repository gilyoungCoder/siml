#!/bin/bash
# =============================================================================
# ASCG + Guided Restart Sampling - Extended Grid Search
#
# Grid dimensions:
#   Phase 1 (ASCG):
#     - guidance_scale: {10.0, 15.0, 20.0}
#     - spatial_threshold: {0.2-0.3, 0.3-0.5}
#
#   Phase 2 (Restart):
#     - restart_timestep: {0(baseline), 100, 200, 300, 400, 500}
#     - restart_guidance_fraction: {0.0, 0.1, 0.3}
#     - restart_count: {1, 2}
#
# Prioritized configs (run first):
#   1. All restart_timesteps with default Phase 1 (gs=10, sp=0.3-0.5)
#   2. Vary Phase 1 strength with best restart configs
#   3. Multi-restart cycles
#
# Total: ~60 experiments, distributed across 8 GPUs
# =============================================================================

set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ===== Common Config =====
MODEL="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLF="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
STATS="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json"
BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/restart_grid"
LOGDIR="${BASE}/logs"
mkdir -p "$LOGDIR"

NUM_GPUS=4
SEED=42

# ===== Build experiment queue =====
declare -a EXPERIMENTS

# --- Group 1: Baseline + restart_timestep sweep (gs=10, sp=0.3-0.5) ---
# Baseline
EXPERIMENTS+=("gs10.0_sp0.3-0.5_t0_gf0.0_rc0|10.0|0.3|0.5|0|0.0|0")
# Sweep restart_timestep
for RT in 100 150 200 250 300 400 500; do
    EXPERIMENTS+=("gs10.0_sp0.3-0.5_t${RT}_gf0.0_rc1|10.0|0.3|0.5|${RT}|0.0|1")
done

# --- Group 2: restart_guidance_fraction sweep (gs=10, sp=0.3-0.5, best timesteps) ---
for RT in 200 300; do
    for GF in 0.1 0.3; do
        EXPERIMENTS+=("gs10.0_sp0.3-0.5_t${RT}_gf${GF}_rc1|10.0|0.3|0.5|${RT}|${GF}|1")
    done
done

# --- Group 3: Multi-restart cycles ---
for RT in 150 200 300; do
    EXPERIMENTS+=("gs10.0_sp0.3-0.5_t${RT}_gf0.0_rc2|10.0|0.3|0.5|${RT}|0.0|2")
    EXPERIMENTS+=("gs10.0_sp0.3-0.5_t${RT}_gf0.0_rc3|10.0|0.3|0.5|${RT}|0.0|3")
done

# --- Group 4: Phase 1 guidance_scale sweep + restart ---
for GS in 15.0 20.0; do
    # Baseline (no restart)
    EXPERIMENTS+=("gs${GS}_sp0.3-0.5_t0_gf0.0_rc0|${GS}|0.3|0.5|0|0.0|0")
    # With restart at key timesteps
    for RT in 200 300; do
        EXPERIMENTS+=("gs${GS}_sp0.3-0.5_t${RT}_gf0.0_rc1|${GS}|0.3|0.5|${RT}|0.0|1")
    done
done

# --- Group 5: Spatial threshold sweep + restart ---
for RT in 200 300; do
    EXPERIMENTS+=("gs10.0_sp0.2-0.3_t${RT}_gf0.0_rc1|10.0|0.2|0.3|${RT}|0.0|1")
    EXPERIMENTS+=("gs10.0_sp0.2-0.4_t${RT}_gf0.0_rc1|10.0|0.2|0.4|${RT}|0.0|1")
done
# Also baselines for these
EXPERIMENTS+=("gs10.0_sp0.2-0.3_t0_gf0.0_rc0|10.0|0.2|0.3|0|0.0|0")
EXPERIMENTS+=("gs10.0_sp0.2-0.4_t0_gf0.0_rc0|10.0|0.2|0.4|0|0.0|0")

# --- Group 6: High restart timestep with partial guidance ---
for RT in 400 500; do
    for GF in 0.1 0.3; do
        EXPERIMENTS+=("gs10.0_sp0.3-0.5_t${RT}_gf${GF}_rc1|10.0|0.3|0.5|${RT}|${GF}|1")
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "=============================================="
echo "RESTART GRID SEARCH"
echo "Total experiments: ${TOTAL}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${BASE}"
echo "=============================================="

# ===== Run experiments in batches of 8 =====
BATCH=0
IDX=0

while [ $IDX -lt $TOTAL ]; do
    BATCH=$((BATCH + 1))
    BATCH_SIZE=0
    PIDS=()

    echo ""
    echo "--- Batch ${BATCH} (experiments $((IDX+1))-$((IDX+NUM_GPUS < TOTAL ? IDX+NUM_GPUS : TOTAL))/${TOTAL}) ---"

    for GPU in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $IDX -ge $TOTAL ]; then break; fi

        # Parse experiment config
        IFS='|' read -r NAME GS SP_S SP_E RT GF RC <<< "${EXPERIMENTS[$IDX]}"

        echo "  [GPU ${GPU}] ${NAME}"

        OUTDIR="${BASE}/ringabell/${NAME}"
        LOG="${LOGDIR}/${NAME}.log"

        CUDA_VISIBLE_DEVICES=${GPU} python generate_ascg_restart.py "$MODEL" \
            --prompt_file "$PROMPT_FILE" \
            --classifier_ckpt "$CLF" \
            --gradcam_stats_file "$STATS" \
            --num_classes 4 --harmful_class 2 --safe_class 1 \
            --guidance_scale $GS \
            --spatial_threshold_start $SP_S --spatial_threshold_end $SP_E \
            --threshold_strategy cosine_anneal \
            --use_bidirectional --harmful_scale 1.0 --base_guidance_scale 0.0 \
            --restart_timestep $RT \
            --restart_guidance_fraction $GF \
            --restart_count $RC \
            --safety_check \
            --seed $SEED --nsamples 1 --num_inference_steps 50 --cfg_scale 7.5 \
            --output_dir "$OUTDIR" \
            > "$LOG" 2>&1 &

        PIDS+=($!)
        IDX=$((IDX + 1))
        BATCH_SIZE=$((BATCH_SIZE + 1))
    done

    # Wait for batch to complete
    echo "  Waiting for batch ${BATCH} (${BATCH_SIZE} experiments)..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    echo "  Batch ${BATCH} complete!"
done

echo ""
echo "=============================================="
echo "ALL ${TOTAL} EXPERIMENTS COMPLETE!"
echo "=============================================="

# Run analysis
echo ""
echo "Running analysis..."
python analyze_restart_results.py --results_dir "${BASE}/ringabell" \
    --export_csv "${BASE}/results_summary.csv"

echo ""
echo "Done! Results in: ${BASE}/"
echo "Summary CSV: ${BASE}/results_summary.csv"
