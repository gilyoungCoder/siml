#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# SD3 Eval: NudeNet + Qwen3-VL for ALL 5 methods × ALL datasets
# Skips already-evaluated directories. Run after all generation is done.
# Usage: nohup bash run_eval_sd3_all.sh GPU_NN GPU_QW > log 2>&1 &
# ============================================================================

GPU_NN=${1:-5}
GPU_QW=${2:-7}

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VLM="/mnt/home3/yhgil99/unlearning/vlm"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"

count_images() { find "$1" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l; }

METHODS="baseline safree safegen safe_denoiser sgf"
NUDITY_DS="rab mma p4dn unlearndiff"

echo "$(date) ===== SD3 FULL EVAL START (NudeNet=GPU$GPU_NN, Qwen=GPU$GPU_QW) ====="

# --- NudeNet ---
echo "$(date) ===== NUDENET ====="
for METHOD in $METHODS; do
    for DS in $NUDITY_DS coco250; do
        DIR="$OUT/$METHOD/$DS"
        N=$(count_images "$DIR")
        [ "$N" -eq 0 ] && continue
        [ -f "$DIR/results_nudenet.txt" ] && { echo "$(date) [SKIP] NudeNet $METHOD/$DS"; continue; }
        echo "$(date) [GPU$GPU_NN] NudeNet $METHOD/$DS ($N imgs)"
        CUDA_VISIBLE_DEVICES=$GPU_NN $PYTHON "$VLM/eval_nudenet.py" "$DIR" --threshold 0.5
    done
done

# --- Qwen3-VL ---
echo "$(date) ===== QWEN3-VL ====="
for METHOD in $METHODS; do
    # Nudity datasets
    for DS in $NUDITY_DS; do
        DIR="$OUT/$METHOD/$DS"
        [ -f "$DIR/categories_qwen3_vl_nudity.json" ] && { echo "$(date) [SKIP] Qwen $METHOD/$DS nudity"; continue; }
        N=$(count_images "$DIR")
        [ "$N" -eq 0 ] && continue
        echo "$(date) [GPU$GPU_QW] Qwen $METHOD/$DS nudity ($N imgs)"
        cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" nudity qwen 2>&1 | tail -3
    done

    # COCO
    DIR="$OUT/$METHOD/coco250"
    if [ ! -f "$DIR/categories_qwen3_vl_nudity.json" ]; then
        N=$(count_images "$DIR")
        if [ "$N" -gt 0 ]; then
            echo "$(date) [GPU$GPU_QW] Qwen $METHOD/coco250 nudity ($N imgs)"
            cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" nudity qwen 2>&1 | tail -3
        fi
    else
        echo "$(date) [SKIP] Qwen $METHOD/coco250 nudity"
    fi

    # MJA datasets
    for pair in "mja_sexual nudity" "mja_violent violence" "mja_disturbing shocking" "mja_illegal illegal"; do
        DS=$(echo $pair | cut -d' ' -f1)
        CONCEPT=$(echo $pair | cut -d' ' -f2)
        DIR="$OUT/$METHOD/$DS"
        [ -f "$DIR/categories_qwen3_vl_${CONCEPT}.json" ] && { echo "$(date) [SKIP] Qwen $METHOD/$DS $CONCEPT"; continue; }
        N=$(count_images "$DIR")
        [ "$N" -eq 0 ] && continue
        echo "$(date) [GPU$GPU_QW] Qwen $METHOD/$DS $CONCEPT ($N imgs)"
        cd "$VLM" && CUDA_VISIBLE_DEVICES=$GPU_QW $VLM_PYTHON opensource_vlm_i2p_all.py "$DIR" "$CONCEPT" qwen 2>&1 | tail -3
    done
done

echo "$(date) ===== SD3 FULL EVAL COMPLETE ====="
