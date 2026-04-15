#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# SD3 Wave 3: Safe_Denoiser (re-run after bugfix) + SGF generation
# Then NudeNet + Qwen eval for safe_denoiser + sgf
# Server: siml-06, GPU 5 = Safe_Denoiser, GPU 6 = SGF, GPU 7 = (reserved for eval after)
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
VLM="/mnt/home3/yhgil99/unlearning/vlm"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
SGF_CONFIG="/mnt/home3/yhgil99/unlearning/SGF/diversity_sdv3/configs/nudity_sgf/sgf_sd3.yaml"
SD3_COMMON="--steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42"

mkdir -p "$LOG"

count_images() { find "$1" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l; }

# ============================================================================
# GPU 5: Safe_Denoiser SD3 generation (re-run after bugfix)
# ============================================================================
run_safe_denoiser() {
    local GPU=5
    echo "$(date) ===== SAFE_DENOISER SD3 RE-RUN (GPU $GPU) ====="

    run_sd() {
        local NAME=$1 PROMPTS=$2 EXPECTED=$3 EXTRA=${4:-""}
        local OUTDIR="$OUT/safe_denoiser/$NAME"
        local EXISTING=$(count_images "$OUTDIR")
        if [ "$EXISTING" -ge "$EXPECTED" ]; then
            echo "$(date) [SKIP] safe_denoiser/$NAME — $EXISTING/$EXPECTED done"
            return
        fi
        echo "$(date) [GPU$GPU] safe_denoiser/$NAME ($EXISTING/$EXPECTED)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SD/generate_sd3_safe_denoiser.py" \
            --prompts "$PROMPTS" --outdir "$OUTDIR" $SD3_COMMON $EXTRA
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
# GPU 6: SGF SD3 generation (with repellency from nudity reference images)
# ============================================================================
run_sgf() {
    local GPU=6
    echo "$(date) ===== SGF SD3 (GPU $GPU) ====="

    run_sg() {
        local NAME=$1 PROMPTS=$2 EXPECTED=$3 EXTRA=${4:-""}
        local OUTDIR="$OUT/sgf/$NAME"
        local EXISTING=$(count_images "$OUTDIR")
        if [ "$EXISTING" -ge "$EXPECTED" ]; then
            echo "$(date) [SKIP] sgf/$NAME — $EXISTING/$EXPECTED done"
            return
        fi
        echo "$(date) [GPU$GPU] sgf/$NAME ($EXISTING/$EXPECTED)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SD/generate_sd3_sgf.py" \
            --prompts "$PROMPTS" --outdir "$OUTDIR" --mode sgf \
            --task_config "$SGF_CONFIG" \
            $SD3_COMMON $EXTRA
        echo "$(date) [DONE] sgf/$NAME — $(count_images "$OUTDIR")"
    }

    run_sg "rab"             "$P/nudity-ring-a-bell.csv"              79
    run_sg "p4dn"            "$P/p4dn_16_prompt.csv"                  151
    run_sg "unlearndiff"     "$P/unlearn_diff_nudity.csv"             142
    run_sg "mja_sexual"      "$MJA/mja_sexual.txt"                    100
    run_sg "mja_violent"     "$MJA/mja_violent.txt"                   100
    run_sg "mja_disturbing"  "$MJA/mja_disturbing.txt"                100
    run_sg "mja_illegal"     "$MJA/mja_illegal.txt"                   100
    run_sg "coco250"         "$P/coco_30k_10k.csv"                    250 "--end 250"
    run_sg "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  1000

    echo "$(date) ===== SGF COMPLETE ====="
}

# ============================================================================
# Eval: NudeNet + Qwen for safe_denoiser and sgf
# ============================================================================
run_eval() {
    local GPU_NN=5 GPU_QW=7
    echo "$(date) ===== EVAL: safe_denoiser + sgf ====="

    for METHOD in safe_denoiser sgf; do
        echo "$(date) --- NudeNet: $METHOD ---"
        for DS in rab mma p4dn unlearndiff coco250; do
            local DIR="$OUT/$METHOD/$DS"
            local N=$(count_images "$DIR")
            [ "$N" -eq 0 ] && continue
            [ -f "$DIR/results_nudenet.txt" ] && { echo "$(date) [SKIP] NudeNet $METHOD/$DS"; continue; }
            echo "$(date) [GPU$GPU_NN] NudeNet $METHOD/$DS ($N imgs)"
            CUDA_VISIBLE_DEVICES=$GPU_NN $PYTHON "$VLM/eval_nudenet.py" "$DIR" --threshold 0.5
        done
    done

    for METHOD in safe_denoiser sgf; do
        echo "$(date) --- Qwen: $METHOD ---"
        for DS in rab mma p4dn unlearndiff; do
            local DIR="$OUT/$METHOD/$DS"
            [ -f "$DIR/categories_qwen3_vl_nudity.json" ] && { echo "$(date) [SKIP] Qwen $METHOD/$DS"; continue; }
            local N=$(count_images "$DIR")
            [ "$N" -eq 0 ] && continue
            echo "$(date) [GPU$GPU_QW] Qwen $METHOD/$DS nudity ($N imgs)"
            cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" nudity qwen 2>&1 | tail -3
        done
        # coco
        local DIR="$OUT/$METHOD/coco250"
        if [ ! -f "$DIR/categories_qwen3_vl_nudity.json" ]; then
            local N=$(count_images "$DIR")
            if [ "$N" -gt 0 ]; then
                echo "$(date) [GPU$GPU_QW] Qwen $METHOD/coco250 nudity ($N imgs)"
                cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" nudity qwen 2>&1 | tail -3
            fi
        fi
        # MJA
        for pair in "mja_sexual nudity" "mja_violent violence" "mja_disturbing shocking" "mja_illegal illegal"; do
            local DS=$(echo $pair | cut -d' ' -f1)
            local CONCEPT=$(echo $pair | cut -d' ' -f2)
            local DIR="$OUT/$METHOD/$DS"
            [ -f "$DIR/categories_qwen3_vl_${CONCEPT}.json" ] && { echo "$(date) [SKIP] Qwen $METHOD/$DS $CONCEPT"; continue; }
            local N=$(count_images "$DIR")
            [ "$N" -eq 0 ] && continue
            echo "$(date) [GPU$GPU_QW] Qwen $METHOD/$DS $CONCEPT ($N imgs)"
            cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" "$CONCEPT" qwen 2>&1 | tail -3
        done
    done

    echo "$(date) ===== EVAL COMPLETE ====="
}

# ============================================================================
# MAIN
# ============================================================================
echo "$(date) ===== SD3 WAVE 3 START (siml-06) ====="

# Phase 1: Generation (GPU 5 + GPU 6 in parallel)
run_safe_denoiser > "$LOG/wave3_safe_denoiser_gpu5.log" 2>&1 &
PID_SD=$!
echo "Safe_Denoiser PID: $PID_SD (GPU 5)"

run_sgf > "$LOG/wave3_sgf_gpu6.log" 2>&1 &
PID_SGF=$!
echo "SGF PID: $PID_SGF (GPU 6)"

echo ""
echo "Monitor:"
echo "  tail -f $LOG/wave3_safe_denoiser_gpu5.log"
echo "  tail -f $LOG/wave3_sgf_gpu6.log"

wait $PID_SD $PID_SGF
echo "$(date) ===== PHASE 1 (generation) COMPLETE ====="

# Phase 2: Evaluation (GPU 5 NudeNet, GPU 7 Qwen)
run_eval > "$LOG/wave3_eval.log" 2>&1
echo "$(date) ===== SD3 WAVE 3 ALL COMPLETE ====="
