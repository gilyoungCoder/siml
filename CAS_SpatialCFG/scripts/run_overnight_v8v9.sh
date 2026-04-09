#!/bin/bash
# ============================================================
# Overnight Master Script: v6/v7 eval + v8/v9 gen+eval
# Runs everything sequentially per GPU to avoid conflicts
# ============================================================
set -e

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
CONCEPT_DIR="$BASE/exemplars/sd14/concept_directions.pt"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COCO_PROMPTS="$BASE/prompts/coco_30.txt"

# IMPORTANT: cd away from source trees to avoid numpy import issues
cd /mnt/home3/yhgil99/unlearning

echo "============================================"
echo "Overnight Script Started: $(date)"
echo "============================================"

# ============================================================
# Helper: run nudenet + qwen eval on a directory
# ============================================================
run_eval() {
    local DIR=$1
    local GPU=$2
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then
        echo "SKIP eval $NAME (dir not found)"
        return
    fi

    # NudeNet
    if [ ! -f "$DIR/results_nudenet.txt" ]; then
        echo "[GPU $GPU] NudeNet: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$DIR" 2>&1 | tail -3
    fi

    # Qwen3-VL
    if [ ! -f "$DIR/results_qwen_nudity.txt" ]; then
        echo "[GPU $GPU] Qwen3-VL: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$EVAL_VLM" "$DIR" nudity qwen 2>&1 | tail -3
    fi
}

# ============================================================
# Helper: run generation + eval
# ============================================================
run_gen_eval() {
    local GEN_SCRIPT=$1
    local OUTDIR=$2
    local GPU=$3
    local NAME=$(basename "$OUTDIR")
    shift 3
    local ARGS="$@"

    # Generate
    local NIMGS=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -ge 316 ]; then
        echo "[GPU $GPU] SKIP gen $NAME ($NIMGS images exist)"
    else
        echo "[GPU $GPU] Generating: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GEN_SCRIPT" $ARGS 2>&1 | tail -5
    fi

    # Eval
    run_eval "$OUTDIR" "$GPU"
}

# ============================================================
# GPU 0: v6 + v7 evaluations (NudeNet + Qwen)
# ============================================================
(
    echo "=== GPU 0: v6/v7 evaluations ==="

    # v6 evals
    run_eval "$BASE/outputs/v6/v6_crossattn_ts10_as15" 0
    run_eval "$BASE/outputs/v6/v6_crossattn_ts15_as15" 0
    run_eval "$BASE/outputs/v6/COCO_v6_crossattn" 0

    # v7 evals
    run_eval "$BASE/outputs/v7/v7_hyb_ts10_as15" 0
    run_eval "$BASE/outputs/v7/v7_hyb_ts15_as15" 0
    run_eval "$BASE/outputs/v7/v7_hyb_ts10_as10" 0
    run_eval "$BASE/outputs/v7/v7_sld_s10" 0
    run_eval "$BASE/outputs/v7/v7_ainp_s10" 0
    run_eval "$BASE/outputs/v7/COCO_v7_hyb_ts10_as15" 0

    echo "=== GPU 0: DONE ==="
) > "$BASE/outputs/overnight_gpu0.log" 2>&1 &
PID0=$!

# ============================================================
# GPU 1: v8 exemplar_proj variants
# ============================================================
(
    echo "=== GPU 1: v8 exemplar_proj variants ==="

    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s5" 1 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_proj_s5" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj --safety_scale 5.0 --cas_threshold 0.6

    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_proj_s15" 1 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_proj_s15" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj --safety_scale 15.0 --cas_threshold 0.6

    echo "=== GPU 1: DONE ==="
) > "$BASE/outputs/overnight_gpu1.log" 2>&1 &
PID1=$!

# ============================================================
# GPU 2: v8 exemplar_proj_dual + exemplar_proj_prompt
# ============================================================
(
    echo "=== GPU 2: v8 dual + proj_prompt ==="

    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_dual_ts15_as15" 2 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_dual_ts15_as15" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj_dual --safety_scale 1.0 \
        --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_projp_s1" 2 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_projp_s1" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj_prompt --safety_scale 1.0 \
        --proj_scale 1.0 --cas_threshold 0.6

    echo "=== GPU 2: DONE ==="
) > "$BASE/outputs/overnight_gpu2.log" 2>&1 &
PID2=$!

# ============================================================
# GPU 3: v9 exemplar_hybrid + exemplar_sld variants
# ============================================================
(
    echo "=== GPU 3: v9 variants ==="

    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts15_as15" 3 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_exhyb_ts15_as15" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_hybrid --safety_scale 1.0 \
        --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exsld_s10" 3 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_exsld_s10" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_sld --safety_scale 10.0 --cas_threshold 0.6

    echo "=== GPU 3: DONE ==="
) > "$BASE/outputs/overnight_gpu3.log" 2>&1 &
PID3=$!

# ============================================================
# GPU 4: v9 contrast + inpaint variants
# ============================================================
(
    echo "=== GPU 4: v9 contrast/inpaint ==="

    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_contrast_s5" 4 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_contrast_s5" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_contrast --safety_scale 5.0 --cas_threshold 0.6

    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_contrast_s10" 4 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_contrast_s10" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_contrast --safety_scale 10.0 --cas_threshold 0.6

    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_inpaint_s1" 4 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_inpaint_s1" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_inpaint --safety_scale 1.0 --cas_threshold 0.6

    echo "=== GPU 4: DONE ==="
) > "$BASE/outputs/overnight_gpu4.log" 2>&1 &
PID4=$!

# ============================================================
# GPU 5: v8_proj_s10 already running → after it finishes, eval + COCO
# ============================================================
(
    echo "=== GPU 5: wait for v8_proj_s10, then eval + COCO ==="

    # Wait for the already-running v8_proj_s10 to finish
    while [ "$(ls "$BASE/outputs/v8/v8_proj_s10"/*.png 2>/dev/null | wc -l)" -lt 316 ]; do
        sleep 60
    done
    echo "v8_proj_s10 generation complete!"

    run_eval "$BASE/outputs/v8/v8_proj_s10" 5

    # COCO FP for v8
    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/COCO_v8_proj_s10" 5 \
        --prompts "$COCO_PROMPTS" --outdir "$BASE/outputs/v8/COCO_v8_proj_s10" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj --safety_scale 10.0 --cas_threshold 0.6

    # v8 hybrid with exemplar anchor
    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_hyb_ts10_as15" 5 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_hyb_ts10_as15" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --safety_scale 1.0 \
        --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    echo "=== GPU 5: DONE ==="
) > "$BASE/outputs/overnight_gpu5.log" 2>&1 &
PID5=$!

# ============================================================
# GPU 6: v8_dual_ts10_as15 already running → after it, eval + more
# ============================================================
(
    echo "=== GPU 6: wait for v8_dual, then eval + more ==="

    while [ "$(ls "$BASE/outputs/v8/v8_dual_ts10_as15"/*.png 2>/dev/null | wc -l)" -lt 316 ]; do
        sleep 60
    done
    echo "v8_dual_ts10_as15 generation complete!"

    run_eval "$BASE/outputs/v8/v8_dual_ts10_as15" 6

    # v8 exemplar_proj_prompt with higher strength
    run_gen_eval "$BASE/generate_v8.py" "$BASE/outputs/v8/v8_projp_s15" 6 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v8/v8_projp_s15" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_proj_prompt --safety_scale 1.0 \
        --proj_scale 1.5 --cas_threshold 0.6

    echo "=== GPU 6: DONE ==="
) > "$BASE/outputs/overnight_gpu6.log" 2>&1 &
PID6=$!

# ============================================================
# GPU 7: v9_exhyb already running → after it, eval + COCO + more
# ============================================================
(
    echo "=== GPU 7: wait for v9_exhyb, then eval + COCO + more ==="

    while [ "$(ls "$BASE/outputs/v9/v9_exhyb_ts10_as15"/*.png 2>/dev/null | wc -l)" -lt 316 ]; do
        sleep 60
    done
    echo "v9_exhyb_ts10_as15 generation complete!"

    run_eval "$BASE/outputs/v9/v9_exhyb_ts10_as15" 7

    # COCO FP for v9
    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/COCO_v9_exhyb" 7 \
        --prompts "$COCO_PROMPTS" --outdir "$BASE/outputs/v9/COCO_v9_exhyb" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_hybrid --safety_scale 1.0 \
        --target_scale 10 --anchor_scale 15 --cas_threshold 0.6

    # v9 exemplar_sld s=5
    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exsld_s5" 7 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_exsld_s5" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_sld --safety_scale 5.0 --cas_threshold 0.6

    # v9 exemplar_hybrid ts=5 as=10
    run_gen_eval "$BASE/generate_v9.py" "$BASE/outputs/v9/v9_exhyb_ts5_as10" 7 \
        --prompts "$PROMPTS" --outdir "$BASE/outputs/v9/v9_exhyb_ts5_as10" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_hybrid --safety_scale 1.0 \
        --target_scale 5 --anchor_scale 10 --cas_threshold 0.6

    echo "=== GPU 7: DONE ==="
) > "$BASE/outputs/overnight_gpu7.log" 2>&1 &
PID7=$!

echo ""
echo "============================================"
echo "All jobs launched!"
echo "  GPU 0: v6/v7 eval (PID $PID0)"
echo "  GPU 1: v8 proj s5/s15 (PID $PID1)"
echo "  GPU 2: v8 dual/projp (PID $PID2)"
echo "  GPU 3: v9 hybrid/sld (PID $PID3)"
echo "  GPU 4: v9 contrast/inpaint (PID $PID4)"
echo "  GPU 5: v8 proj_s10 eval + COCO + hybrid (PID $PID5)"
echo "  GPU 6: v8 dual eval + projp_s15 (PID $PID6)"
echo "  GPU 7: v9 exhyb eval + COCO + more (PID $PID7)"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  tail -f $BASE/outputs/overnight_gpu*.log"
echo "  nvidia-smi"
echo ""

# Wait for everything
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

echo ""
echo "============================================"
echo "ALL OVERNIGHT JOBS COMPLETE! $(date)"
echo "============================================"

# Print summary
echo ""
echo "=== V8 Results ==="
for d in "$BASE"/outputs/v8/v8_*/; do
    name=$(basename "$d")
    nn="-"; [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$d/results_nudenet.txt" 2>/dev/null || echo "-")
    sr="-"
    if [ -f "$d/results_qwen_nudity.txt" ]; then
        s=$(grep -oP 'Safe: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        p=$(grep -oP 'Partial: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        nr=$(grep -oP 'NotRel: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        f=$(grep -oP 'Full: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        t=$((s+p+nr+f))
        [ "$t" -gt 0 ] && sr=$(echo "scale=1; ($s+$p)*100/$t" | bc 2>/dev/null || echo "-")
    fi
    printf "%-30s NN=%-8s SR=%-8s\n" "$name" "$nn" "$sr"
done

echo ""
echo "=== V9 Results ==="
for d in "$BASE"/outputs/v9/v9_*/; do
    name=$(basename "$d")
    nn="-"; [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$d/results_nudenet.txt" 2>/dev/null || echo "-")
    sr="-"
    if [ -f "$d/results_qwen_nudity.txt" ]; then
        s=$(grep -oP 'Safe: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        p=$(grep -oP 'Partial: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        nr=$(grep -oP 'NotRel: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        f=$(grep -oP 'Full: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
        t=$((s+p+nr+f))
        [ "$t" -gt 0 ] && sr=$(echo "scale=1; ($s+$p)*100/$t" | bc 2>/dev/null || echo "-")
    fi
    printf "%-30s NN=%-8s SR=%-8s\n" "$name" "$nn" "$sr"
done
