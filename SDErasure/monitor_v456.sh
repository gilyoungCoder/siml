#!/bin/bash
# Monitor v4, v5, v6 training, run evaluation when each finishes
set -e

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"

MODEL_ID="CompVis/stable-diffusion-v1-4"
RINGABELL_PROMPTS="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
VLM_EVAL_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
GEN_SCRIPT="/mnt/home/yhgil99/unlearning/SDErasure/generate_from_prompts.py"
BASE_OUT="/mnt/home/yhgil99/unlearning/SDErasure/outputs"

V4_PID=$1
V5_PID=$2
V6_PID=$3

# Use GPUs 0 and 7 for gen/eval (free GPUs)
GEN_GPU=0
EVAL_GPU=7

echo "Monitoring v4 (PID $V4_PID), v5 (PID $V5_PID), v6 (PID $V6_PID)"
echo "Gen GPU: $GEN_GPU, Eval GPU: $EVAL_GPU"

run_eval() {
    local version=$1
    local unet_dir=$2
    local gen_dir=$3

    echo ""
    echo "============================================================"
    echo "[$version] Training complete! Running evaluation..."
    echo "============================================================"

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

    echo "[$version] Running Qwen3-VL evaluation..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_EVAL_SCRIPT" "$gen_dir" nudity qwen

    echo ""
    echo "[$version] RESULTS:"
    cat "$gen_dir/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (no results file)"
    echo ""
}

v4_done=false
v5_done=false
v6_done=false

while true; do
    if ! $v4_done && ! kill -0 $V4_PID 2>/dev/null; then
        v4_done=true
        echo ">>> v4 training finished! <<<"
        run_eval "v4" "$BASE_OUT/sderasure_nudity_v4/unet" "$BASE_OUT/sderasure_nudity_v4/ringabell_images"
    fi

    if ! $v5_done && ! kill -0 $V5_PID 2>/dev/null; then
        v5_done=true
        echo ">>> v5 training finished! <<<"
        run_eval "v5" "$BASE_OUT/sderasure_nudity_v5/unet" "$BASE_OUT/sderasure_nudity_v5/ringabell_images"
    fi

    if ! $v6_done && ! kill -0 $V6_PID 2>/dev/null; then
        v6_done=true
        echo ">>> v6 training finished! <<<"
        run_eval "v6" "$BASE_OUT/sderasure_nudity_v6/unet" "$BASE_OUT/sderasure_nudity_v6/ringabell_images"
    fi

    if $v4_done && $v5_done && $v6_done; then
        break
    fi

    sleep 30
done

echo ""
echo "============================================================"
echo "ALL RESULTS COMPARISON"
echo "============================================================"
echo ""
echo "--- Baseline (SD v1.4) ---"
cat "$BASE_OUT/baseline_sd14/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
echo ""
echo "--- v3 (eta=5, attn-only) ---"
cat "$BASE_OUT/sderasure_nudity_v3/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
echo ""
echo "--- v4 (eta=10, lr=1e-4, attn-only) ---"
cat "$BASE_OUT/sderasure_nudity_v4/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
echo ""
echo "--- v5 (eta=8, lr=5e-5, full UNet) ---"
cat "$BASE_OUT/sderasure_nudity_v5/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
echo ""
echo "--- v6 (eta=15, attn-only) ---"
cat "$BASE_OUT/sderasure_nudity_v6/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
echo ""
echo "--- SGF (target) ---"
echo "SR: 74.7%"
echo "============================================================"
