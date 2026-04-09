#!/bin/bash
# ============================================================
# Extra Grid Search: GPU 4-7, 더 넓은 하이퍼파라미터 탐색
# ============================================================
set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COCO_PROMPTS="$BASE/prompts/coco_30.txt"
CONCEPT_DIR="$BASE/exemplars/sd14/concept_directions.pt"

cd /tmp

echo "============================================"
echo "Extra Grid Search (GPU 4-7): $(date)"
echo "============================================"

run_full() {
    local GPU=$1
    local GEN_SCRIPT=$2
    local OUTDIR=$3
    local PROMPT_FILE=$4
    shift 4
    local ARGS="$@"
    local NAME=$(basename "$OUTDIR")
    local MIN_IMGS=316
    echo "$NAME" | grep -qi "coco" && MIN_IMGS=120

    local NIMGS=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -ge "$MIN_IMGS" ]; then
        echo "[GPU $GPU] SKIP gen $NAME ($NIMGS imgs)"
    else
        echo "[GPU $GPU] GEN: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GEN_SCRIPT" \
            --prompts "$PROMPT_FILE" --outdir "$OUTDIR" $ARGS 2>&1 | tail -3
    fi

    if [ -f "$OUTDIR/results_nudenet.txt" ]; then
        echo "[GPU $GPU] SKIP NN $NAME"
    elif [ -d "$OUTDIR" ]; then
        echo "[GPU $GPU] NN: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$OUTDIR" 2>&1 | tail -3
    fi

    if [ -f "$OUTDIR/results_qwen_nudity.txt" ] || [ -f "$OUTDIR/results_qwen3_vl_nudity.txt" ]; then
        echo "[GPU $GPU] SKIP Qwen $NAME"
    elif [ -d "$OUTDIR" ]; then
        echo "[GPU $GPU] Qwen: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_VLM" "$OUTDIR" nudity qwen 2>&1 | tail -3
    fi

    echo "[GPU $GPU] COMPLETE: $NAME"
    echo ""
}

# ============================================================
# GPU 4: v7 CAS threshold sweep + extreme scales
# ============================================================
(
    echo "=== GPU 4: v7 CAS sweep + extreme ==="
    G=4

    # CAS threshold variations (best config ts15_as15 기반)
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_cas05" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.5

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_cas07" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.7

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_cas04" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.4

    # Extreme scales
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts25_as25" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 25 --anchor_scale 25 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts30_as20" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 30 --anchor_scale 20 --cas_threshold 0.6

    # SLD extreme
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_sld_s20" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode sld --safety_scale 20.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_sld_s25" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode sld --safety_scale 25.0 --cas_threshold 0.6

    # anchor_inpaint sweep
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_ainp_s03" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode anchor_inpaint --safety_scale 0.3 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_ainp_s05" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode anchor_inpaint --safety_scale 0.5 --cas_threshold 0.6

    echo "=== GPU 4: DONE ==="
) > "$BASE/outputs/gridsearch_gpu4.log" 2>&1 &
PID4=$!

# ============================================================
# GPU 5: v7 spatial threshold sweep + asymmetric scales
# ============================================================
(
    echo "=== GPU 5: v7 spatial sweep + asymmetric ==="
    G=5

    # Spatial threshold variations
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_spat02" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --spatial_threshold 0.2

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_spat04" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --spatial_threshold 0.4

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_spat05" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --spatial_threshold 0.5

    # Asymmetric: strong target, weak anchor
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts20_as10" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 10 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts25_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 25 --anchor_scale 15 --cas_threshold 0.6

    # Weak target, strong anchor
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts10_as20" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 20 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts5_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 5 --anchor_scale 15 --cas_threshold 0.6

    # COCO for best new configs
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/COCO_v7_hyb_ts20_as20" "$COCO_PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 20 --cas_threshold 0.6

    echo "=== GPU 5: DONE ==="
) > "$BASE/outputs/gridsearch_gpu5.log" 2>&1 &
PID5=$!

# ============================================================
# GPU 6: v6 extended grid + v7 sigmoid/blur sweep
# ============================================================
(
    echo "=== GPU 6: v6 extended + v7 mask params ==="
    G=6

    # v6 anchor_inpaint + projection
    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ainp_s1" "$PROMPTS" \
        --guide_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ainp_s07" "$PROMPTS" \
        --guide_mode anchor_inpaint --safety_scale 0.7 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_proj_s1" "$PROMPTS" \
        --guide_mode projection --safety_scale 1.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ts10_as10" "$PROMPTS" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 10 --cas_threshold 0.6

    # v7 sigmoid alpha sweep (mask sharpness)
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_sig5" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --sigmoid_alpha 5.0

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_sig20" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --sigmoid_alpha 20.0

    # v7 blur sigma sweep
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_blur0" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --blur_sigma 0.0

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_blur2" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --blur_sigma 2.0

    echo "=== GPU 6: DONE ==="
) > "$BASE/outputs/gridsearch_gpu6.log" 2>&1 &
PID6=$!

# ============================================================
# GPU 7: v8 extreme + v9 extreme + v7 guide_start_frac sweep
# ============================================================
(
    echo "=== GPU 7: v8/v9 extreme + v7 timing ==="
    G=7

    # v8 extreme scales
    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s30" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj --safety_scale 30.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_dual_ts25_as25" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_dual --target_scale 25 --anchor_scale 25 --cas_threshold 0.6

    # v9 extreme scales
    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exsld_s20" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_sld --safety_scale 20.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts25_as25" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 25 --anchor_scale 25 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_contrast_s15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_contrast --safety_scale 15.0 --cas_threshold 0.6

    # v7 guide_start_frac sweep (guidance timing)
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_start02" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --guide_start_frac 0.2

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15_start04" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 \
        --guide_start_frac 0.4

    echo "=== GPU 7: DONE ==="
) > "$BASE/outputs/gridsearch_gpu7.log" 2>&1 &
PID7=$!

echo ""
echo "============================================"
echo "Extra Grid Search Launched!"
echo "  GPU 4: v7 CAS sweep + extreme (PID $PID4)"
echo "  GPU 5: v7 spatial + asymmetric (PID $PID5)"
echo "  GPU 6: v6 extended + v7 mask params (PID $PID6)"
echo "  GPU 7: v8/v9 extreme + v7 timing (PID $PID7)"
echo "============================================"

wait $PID4 $PID5 $PID6 $PID7

echo ""
echo "ALL EXTRA GRID SEARCH COMPLETE: $(date)"
