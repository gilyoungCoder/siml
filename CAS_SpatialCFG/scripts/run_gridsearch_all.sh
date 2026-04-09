#!/bin/bash
# ============================================================
# Master Grid Search: v6/v7/v8/v9 전체 + NudeNet + Qwen eval
# GPU 0-3 사용, 각 GPU에서 순차적으로 gen→NN→Qwen 체인
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

cd /tmp  # avoid numpy source dir issues

echo "============================================"
echo "Grid Search Started: $(date)"
echo "============================================"

# ============================================================
# Helper: generate + nudenet + qwen for one config
# ============================================================
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

    # Generate (skip if done)
    local NIMGS=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -ge "$MIN_IMGS" ]; then
        echo "[GPU $GPU] SKIP gen $NAME ($NIMGS imgs)"
    else
        echo "[GPU $GPU] GEN: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GEN_SCRIPT" \
            --prompts "$PROMPT_FILE" --outdir "$OUTDIR" $ARGS 2>&1 | tail -3
    fi

    # NudeNet (skip if done)
    if [ -f "$OUTDIR/results_nudenet.txt" ]; then
        echo "[GPU $GPU] SKIP NN $NAME"
    elif [ -d "$OUTDIR" ]; then
        echo "[GPU $GPU] NN: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$OUTDIR" 2>&1 | tail -3
    fi

    # Qwen (skip if done — check both filename variants)
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
# Wait for GPU to be free (< 1GB used)
# ============================================================
wait_gpu_free() {
    local GPU=$1
    while true; do
        local MEM=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ "$MEM" -lt 1000 ] 2>/dev/null; then
            break
        fi
        echo "  Waiting for GPU $GPU to be free (${MEM}MiB used)..."
        sleep 60
    done
}

# ============================================================
# GPU 0: v7 grid search (the winner — most configs here)
# ============================================================
(
    echo "=== GPU 0: v7 Grid Search ==="
    G=0

    # --- Already done, just eval ---
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_ainp_s10" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts10_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    # --- New v7 configs ---
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts20_as20" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 20 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts20_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts15_as10" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 10 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_sld_s5" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode sld --safety_scale 5.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_sld_s15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode sld --safety_scale 15.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_ainp_s07" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode anchor_inpaint --safety_scale 0.7 --cas_threshold 0.6

    # --- COCO FP for best configs ---
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/COCO_v7_hyb_ts15_as15" "$COCO_PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    echo "=== GPU 0: DONE ==="
) > "$BASE/outputs/gridsearch_gpu0.log" 2>&1 &
PID0=$!

# ============================================================
# GPU 1: v7 remaining + v6 grid search (wait for current Qwen)
# ============================================================
(
    echo "=== GPU 1: v7 rest + v6 ==="
    G=1
    wait_gpu_free $G

    # v7 remaining eval
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hyb_ts10_as10" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 10 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_sld_s10" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode sld --safety_scale 10.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/COCO_v7_hyb_ts10_as15" "$COCO_PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    # --- v7 new: projection mode ---
    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_proj_s1" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode projection --safety_scale 1.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v7.py" "$BASE/outputs/v7/v7_hybproj_ts10_as15" "$PROMPTS" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid_proj --target_scale 10 --anchor_scale 15 --proj_scale 1.0 --cas_threshold 0.6

    # --- v6 grid search ---
    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ts10_as15" "$PROMPTS" \
        --guide_mode hybrid --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ts15_as15" "$PROMPTS" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ts20_as20" "$PROMPTS" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 20 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_ts20_as15" "$PROMPTS" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_sld_s10" "$PROMPTS" \
        --guide_mode sld --safety_scale 10.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/v6_crossattn_sld_s15" "$PROMPTS" \
        --guide_mode sld --safety_scale 15.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v6.py" "$BASE/outputs/v6/COCO_v6_crossattn" "$COCO_PROMPTS" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    echo "=== GPU 1: DONE ==="
) > "$BASE/outputs/gridsearch_gpu1.log" 2>&1 &
PID1=$!

# ============================================================
# GPU 2: v8 grid search (wait for current Qwen)
# ============================================================
(
    echo "=== GPU 2: v8 Grid Search ==="
    G=2
    wait_gpu_free $G

    # Already done — just eval
    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s10" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj --safety_scale 10.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s5" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj --safety_scale 5.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_dual_ts10_as15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_dual --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_dual_ts15_as15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_dual --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_projp_s15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_prompt --proj_scale 1.5 --cas_threshold 0.6

    # New v8 configs
    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj --safety_scale 15.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s20" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj --safety_scale 20.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_dual_ts20_as20" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_dual --target_scale 20 --anchor_scale 20 --cas_threshold 0.6

    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_projp_s1" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_prompt --proj_scale 1.0 --cas_threshold 0.6

    # COCO FP
    run_full $G "$BASE/generate_v8.py" "$BASE/outputs/v8/COCO_v8_dual_ts15_as15" "$COCO_PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_proj_dual --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    echo "=== GPU 2: DONE ==="
) > "$BASE/outputs/gridsearch_gpu2.log" 2>&1 &
PID2=$!

# ============================================================
# GPU 3: v9 grid search (wait for current Qwen)
# ============================================================
(
    echo "=== GPU 3: v9 Grid Search ==="
    G=3
    wait_gpu_free $G

    # Already done — just eval
    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts10_as15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts15_as15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_contrast_s5" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_contrast --safety_scale 5.0 --cas_threshold 0.6

    # New v9 configs
    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts20_as20" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 20 --anchor_scale 20 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exsld_s10" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_sld --safety_scale 10.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exsld_s15" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_sld --safety_scale 15.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_contrast_s10" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_contrast --safety_scale 10.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_inpaint_s1" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_inpaint --safety_scale 1.0 --cas_threshold 0.6

    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts5_as10" "$PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 5 --anchor_scale 10 --cas_threshold 0.6

    # COCO FP
    run_full $G "$BASE/generate_v9.py" "$BASE/outputs/v9/COCO_v9_exhyb_ts15_as15" "$COCO_PROMPTS" \
        --concept_dir_path "$CONCEPT_DIR" --guide_mode exemplar_hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    echo "=== GPU 3: DONE ==="
) > "$BASE/outputs/gridsearch_gpu3.log" 2>&1 &
PID3=$!

echo ""
echo "============================================"
echo "Grid Search Launched!"
echo "  GPU 0: v7 grid (PID $PID0) — starts immediately"
echo "  GPU 1: v7 rest + v6 grid (PID $PID1) — waits for Qwen"
echo "  GPU 2: v8 grid (PID $PID2) — waits for Qwen"
echo "  GPU 3: v9 grid (PID $PID3) — waits for Qwen"
echo "============================================"
echo ""
echo "Monitor:"
echo "  tail -f $BASE/outputs/gridsearch_gpu*.log"
echo ""

# Wait for all
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "============================================"
echo "ALL GRID SEARCH COMPLETE: $(date)"
echo "============================================"
echo ""

# ============================================================
# Final Summary
# ============================================================
printf "%-35s | %8s | %8s | %8s\n" "Experiment" "NN%" "SR%" "Full%"
printf "%s\n" "$(printf '%.0s-' {1..75})"

for ver in v6 v7 v8 v9; do
    for d in "$BASE"/outputs/$ver/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [[ "$name" == *debug* ]] && continue

        nn="-"
        [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$d/results_nudenet.txt" 2>/dev/null || echo "-")

        sr="-"; full="-"
        QFILE="$d/results_qwen3_vl_nudity.txt"
        [ ! -f "$QFILE" ] && QFILE="$d/results_qwen_nudity.txt"
        if [ -f "$QFILE" ]; then
            s=$(grep -oP 'Safe: \K\d+' "$QFILE" 2>/dev/null || echo 0)
            p=$(grep -oP 'Partial: \K\d+' "$QFILE" 2>/dev/null || echo 0)
            nr=$(grep -oP 'NotRel: \K\d+' "$QFILE" 2>/dev/null || echo 0)
            f=$(grep -oP 'Full: \K\d+' "$QFILE" 2>/dev/null || echo 0)
            t=$((s+p+nr+f))
            if [ "$t" -gt 0 ]; then
                sr=$(echo "scale=1; ($s+$p)*100/$t" | bc 2>/dev/null || echo "-")
                full=$(echo "scale=1; $f*100/$t" | bc 2>/dev/null || echo "-")
            fi
        fi

        printf "%-35s | %8s | %8s | %8s\n" "$ver/$name" "$nn" "$sr" "$full"
    done
done
