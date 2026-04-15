#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# SD3 Wave 2: Safe_Denoiser generation + NudeNet/Qwen eval for all methods
# Server: siml-06, GPUs: 5, 6, 7
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
VLM="/mnt/home3/yhgil99/unlearning/vlm"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
COMMON="--steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42"

mkdir -p "$LOG"

count_images() { ls "$1"/*.png 2>/dev/null | wc -l; }

# ============================================================================
# GPU 5: Safe_Denoiser SD3 generation
# ============================================================================
run_safe_denoiser() {
    local GPU=5
    echo "$(date) ===== SAFE_DENOISER SD3 (GPU $GPU) ====="

    run_sd() {
        local NAME=$1 PROMPTS=$2 EXPECTED=$3 EXTRA=${4:-""}
        local OUTDIR="$OUT/safe_denoiser/$NAME"
        local EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
        if [ "$EXISTING" -ge "$EXPECTED" ]; then
            echo "$(date) [SKIP] safe_denoiser/$NAME — $EXISTING/$EXPECTED done"
            return
        fi
        echo "$(date) [GPU$GPU] safe_denoiser/$NAME ($EXISTING/$EXPECTED)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SD/generate_sd3_safe_denoiser.py" \
            --prompts "$PROMPTS" --outdir "$OUTDIR" $COMMON $EXTRA
        echo "$(date) [DONE] safe_denoiser/$NAME — $(count_images "$OUTDIR")"
    }

    run_sd "rab"             "$P/nudity-ring-a-bell.csv"              79
    run_sd "p4dn"            "$P/p4dn_16_prompt.csv"                  151
    run_sd "unlearndiff"     "$P/unlearn_diff_nudity.csv"             142
    run_sd "mja_sexual"      "$MJA/mja_sexual.txt"                    100
    run_sd "mja_violent"     "$MJA/mja_violent.txt"                   100
    run_sd "mja_disturbing"  "$MJA/mja_disturbing.txt"                100
    run_sd "mja_illegal"     "$MJA/mja_illegal.txt"                   100
    run_sd "coco250"         "$P/coco_30k_10k.csv"                    250 "--end 250"
    run_sd "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  1000

    echo "$(date) ===== SAFE_DENOISER COMPLETE ====="
}

# ============================================================================
# GPU 6: NudeNet evaluation on ALL methods
# ============================================================================
run_nudenet_eval() {
    local GPU=6
    echo "$(date) ===== NUDENET EVAL (GPU $GPU) ====="

    eval_dir() {
        local DIR=$1
        local RESULT="$DIR/results_nudenet.txt"
        if [ -f "$RESULT" ]; then
            echo "$(date) [SKIP] NudeNet: $DIR (already done)"
            return
        fi
        local N=$(count_images "$DIR" 2>/dev/null || echo 0)
        if [ "$N" -eq 0 ]; then
            echo "$(date) [SKIP] NudeNet: $DIR (no images)"
            return
        fi
        echo "$(date) [GPU$GPU] NudeNet: $DIR ($N imgs)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$VLM/eval_nudenet.py" "$DIR" --threshold 0.5
    }

    NUDITY_DATASETS="rab mma p4dn unlearndiff"

    for METHOD in baseline safree safegen; do
        echo "$(date) --- NudeNet: $METHOD ---"
        for DS in $NUDITY_DATASETS; do
            eval_dir "$OUT/$METHOD/$DS"
        done
        eval_dir "$OUT/$METHOD/coco250"
    done

    echo "$(date) ===== NUDENET EVAL COMPLETE ====="
}

# ============================================================================
# GPU 7: Qwen3-VL evaluation on ALL methods
# ============================================================================
run_qwen_eval() {
    local GPU=7
    echo "$(date) ===== QWEN3-VL EVAL (GPU $GPU) ====="

    eval_qwen() {
        local DIR=$1 CONCEPT=$2
        local JSON_FILE="$DIR/categories_qwen3_vl_${CONCEPT}.json"
        if [ -f "$JSON_FILE" ]; then
            echo "$(date) [SKIP] Qwen: $DIR ($CONCEPT already done)"
            return
        fi
        local N=$(count_images "$DIR" 2>/dev/null || echo 0)
        if [ "$N" -eq 0 ]; then
            echo "$(date) [SKIP] Qwen: $DIR (no images)"
            return
        fi
        echo "$(date) [GPU$GPU] Qwen: $DIR $CONCEPT ($N imgs)"
        cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" "$CONCEPT" qwen 2>&1 | tail -3
    }

    for METHOD in baseline safree safegen; do
        echo "$(date) --- Qwen: $METHOD ---"
        for DS in rab mma p4dn unlearndiff; do
            eval_qwen "$OUT/$METHOD/$DS" "nudity"
        done
        eval_qwen "$OUT/$METHOD/coco250" "nudity"
        eval_qwen "$OUT/$METHOD/mja_sexual"      "nudity"
        eval_qwen "$OUT/$METHOD/mja_violent"      "violence"
        eval_qwen "$OUT/$METHOD/mja_disturbing"   "shocking"
        eval_qwen "$OUT/$METHOD/mja_illegal"      "illegal"
    done

    echo "$(date) ===== QWEN3-VL EVAL COMPLETE ====="
}

# ============================================================================
# Safe_Denoiser NudeNet + Qwen (runs after generation completes)
# ============================================================================
run_safe_denoiser_eval() {
    local GPU_NN=6 GPU_QW=7
    echo "$(date) ===== SAFE_DENOISER EVAL ====="

    # NudeNet
    for DS in rab mma p4dn unlearndiff coco250; do
        local DIR="$OUT/safe_denoiser/$DS"
        local N=$(count_images "$DIR" 2>/dev/null || echo 0)
        if [ "$N" -eq 0 ]; then continue; fi
        if [ ! -f "$DIR/results_nudenet.txt" ]; then
            echo "$(date) [GPU$GPU_NN] NudeNet safe_denoiser/$DS ($N imgs)"
            CUDA_VISIBLE_DEVICES=$GPU_NN $PYTHON "$VLM/eval_nudenet.py" "$DIR" --threshold 0.5
        fi
    done

    # Qwen
    for DS in rab mma p4dn unlearndiff; do
        local DIR="$OUT/safe_denoiser/$DS"
        if [ ! -f "$DIR/categories_qwen3_vl_nudity.json" ]; then
            local N=$(count_images "$DIR" 2>/dev/null || echo 0)
            [ "$N" -eq 0 ] && continue
            echo "$(date) [GPU$GPU_QW] Qwen safe_denoiser/$DS ($N imgs)"
            cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" nudity qwen 2>&1 | tail -3
        fi
    done
    # coco
    if [ ! -f "$OUT/safe_denoiser/coco250/categories_qwen3_vl_nudity.json" ]; then
        cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$OUT/safe_denoiser/coco250" nudity qwen 2>&1 | tail -3
    fi
    # MJA
    for pair in "mja_sexual nudity" "mja_violent violence" "mja_disturbing shocking" "mja_illegal illegal"; do
        local DS=$(echo $pair | cut -d' ' -f1)
        local CONCEPT=$(echo $pair | cut -d' ' -f2)
        local DIR="$OUT/safe_denoiser/$DS"
        if [ ! -f "$DIR/categories_qwen3_vl_${CONCEPT}.json" ]; then
            local N=$(count_images "$DIR" 2>/dev/null || echo 0)
            [ "$N" -eq 0 ] && continue
            echo "$(date) [GPU$GPU_QW] Qwen safe_denoiser/$DS $CONCEPT ($N imgs)"
            cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" "$CONCEPT" qwen 2>&1 | tail -3
        fi
    done

    echo "$(date) ===== SAFE_DENOISER EVAL COMPLETE ====="
}

# ============================================================================
# MAIN: Launch GPU 5/6/7 in parallel, then eval safe_denoiser
# ============================================================================
echo "$(date) ===== SD3 WAVE 2 START (siml-06, GPUs 5,6,7) ====="

# GPU 5: Safe_Denoiser generation
run_safe_denoiser > "$LOG/wave2_safe_denoiser_gpu5.log" 2>&1 &
PID_SD=$!
echo "Safe_Denoiser PID: $PID_SD (GPU 5)"

# GPU 6: NudeNet eval (baseline/safree/safegen — already generated)
run_nudenet_eval > "$LOG/wave2_nudenet_gpu6.log" 2>&1 &
PID_NN=$!
echo "NudeNet PID: $PID_NN (GPU 6)"

# GPU 7: Qwen eval (baseline/safree/safegen — already generated)
run_qwen_eval > "$LOG/wave2_qwen_gpu7.log" 2>&1 &
PID_QW=$!
echo "Qwen PID: $PID_QW (GPU 7)"

echo ""
echo "Monitor logs:"
echo "  tail -f $LOG/wave2_safe_denoiser_gpu5.log"
echo "  tail -f $LOG/wave2_nudenet_gpu6.log"
echo "  tail -f $LOG/wave2_qwen_gpu7.log"

wait $PID_SD $PID_NN $PID_QW

echo ""
echo "$(date) ===== PHASE 1 COMPLETE (gen+eval for baseline/safree/safegen) ====="
echo "$(date) ===== PHASE 2: Safe_Denoiser eval ====="

run_safe_denoiser_eval > "$LOG/wave2_safe_denoiser_eval.log" 2>&1

echo "$(date) ===== SD3 WAVE 2 ALL COMPLETE ====="
