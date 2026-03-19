#!/bin/bash
# Monitor v2 and v3 training, run evaluation when each finishes
set -e

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"

MODEL_ID="CompVis/stable-diffusion-v1-4"
RINGABELL_PROMPTS="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
VLM_EVAL_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
GEN_SCRIPT="/mnt/home/yhgil99/unlearning/SDErasure/generate_from_prompts.py"
BASE_OUT="/mnt/home/yhgil99/unlearning/SDErasure/outputs"

V2_PID=$1
V3_PID=$2
GEN_GPU=${3:-2}
EVAL_GPU=${4:-3}

echo "Monitoring v2 (PID $V2_PID) and v3 (PID $V3_PID)"
echo "Will generate on GPU $GEN_GPU, evaluate on GPU $EVAL_GPU"

# Function to run eval for a version
run_eval() {
    local version=$1
    local unet_dir=$2
    local gen_dir=$3

    echo ""
    echo "============================================================"
    echo "[$version] Training complete! Running evaluation..."
    echo "============================================================"

    # Generate Ring-A-Bell images
    if [ ! -d "$gen_dir" ] || [ $(ls "$gen_dir"/*.png 2>/dev/null | wc -l) -lt 79 ]; then
        echo "[$version] Generating Ring-A-Bell images..."
        CUDA_VISIBLE_DEVICES=$GEN_GPU eval $PYTHON "$GEN_SCRIPT" \
            --model_id "$MODEL_ID" \
            --unet_dir "$unet_dir" \
            --prompt_file "$RINGABELL_PROMPTS" \
            --output_dir "$gen_dir" \
            --seed 42 \
            --num_inference_steps 50 \
            --guidance_scale 7.5
        echo "[$version] Generated $(ls "$gen_dir"/*.png | wc -l) images"
    else
        echo "[$version] Images already exist: $(ls "$gen_dir"/*.png | wc -l)"
    fi

    # VLM evaluation
    echo "[$version] Running Qwen3-VL evaluation..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_EVAL_SCRIPT" "$gen_dir" nudity qwen

    echo ""
    echo "[$version] RESULTS:"
    cat "$gen_dir/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (no results file)"
    echo ""
}

# Wait for v3 first (faster, attn-only)
v3_done=false
v2_done=false

while true; do
    if ! $v3_done && ! kill -0 $V3_PID 2>/dev/null; then
        v3_done=true
        echo ""
        echo ">>> v3 training finished! <<<"
        run_eval "v3" "$BASE_OUT/sderasure_nudity_v3/unet" "$BASE_OUT/sderasure_nudity_v3/ringabell_images"
    fi

    if ! $v2_done && ! kill -0 $V2_PID 2>/dev/null; then
        v2_done=true
        echo ""
        echo ">>> v2 training finished! <<<"
        run_eval "v2" "$BASE_OUT/sderasure_nudity_v2/unet" "$BASE_OUT/sderasure_nudity_v2/ringabell_images"
    fi

    if $v2_done && $v3_done; then
        break
    fi

    sleep 30
done

# Final comparison
echo ""
echo "============================================================"
echo "FINAL COMPARISON"
echo "============================================================"
echo ""
echo "--- Baseline (SD v1.4) ---"
cat "$BASE_OUT/baseline_sd14/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (not available)"
echo ""
echo "--- SDErasure v2 (eta=3.0, lr=2e-5, full UNet) ---"
cat "$BASE_OUT/sderasure_nudity_v2/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (not available)"
echo ""
echo "--- SDErasure v3 (eta=5.0, lr=5e-5, attn-only) ---"
cat "$BASE_OUT/sderasure_nudity_v3/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (not available)"
echo ""
echo "--- SGF (target) ---"
echo "SR: 74.7%"
echo "============================================================"
