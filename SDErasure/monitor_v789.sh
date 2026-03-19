#!/bin/bash
set -e

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"

MODEL_ID="CompVis/stable-diffusion-v1-4"
RINGABELL_PROMPTS="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
VLM_EVAL_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
GEN_SCRIPT="/mnt/home/yhgil99/unlearning/SDErasure/generate_from_prompts.py"
BASE_OUT="/mnt/home/yhgil99/unlearning/SDErasure/outputs"

V7_PID=$1
V8_PID=$2
V9_PID=$3
GEN_GPU=1
EVAL_GPU=6

echo "Monitoring v7 (PID $V7_PID), v8 (PID $V8_PID), v9 (PID $V9_PID)"

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
    fi

    echo "[$version] Running Qwen3-VL evaluation..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_EVAL_SCRIPT" "$gen_dir" nudity qwen

    echo ""
    echo "[$version] RESULTS:"
    cat "$gen_dir/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (no results file)"
    echo ""
}

v7_done=false; v8_done=false; v9_done=false

while true; do
    if ! $v7_done && ! kill -0 $V7_PID 2>/dev/null; then
        v7_done=true
        echo ">>> v7 training finished! <<<"
        run_eval "v7" "$BASE_OUT/sderasure_nudity_v7/unet" "$BASE_OUT/sderasure_nudity_v7/ringabell_images"
    fi
    if ! $v8_done && ! kill -0 $V8_PID 2>/dev/null; then
        v8_done=true
        echo ">>> v8 training finished! <<<"
        run_eval "v8" "$BASE_OUT/sderasure_nudity_v8/unet" "$BASE_OUT/sderasure_nudity_v8/ringabell_images"
    fi
    if ! $v9_done && ! kill -0 $V9_PID 2>/dev/null; then
        v9_done=true
        echo ">>> v9 training finished! <<<"
        run_eval "v9" "$BASE_OUT/sderasure_nudity_v9/unet" "$BASE_OUT/sderasure_nudity_v9/ringabell_images"
    fi
    if $v7_done && $v8_done && $v9_done; then break; fi
    sleep 30
done

echo ""
echo "============================================================"
echo "ALL ANCHOR-BASED RESULTS"
echo "============================================================"
for v in v7 v8 v9; do
    echo "--- $v ---"
    cat "$BASE_OUT/sderasure_nudity_${v}/ringabell_images/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR"
done
echo "--- SGF target: SR 74.7% ---"
echo "============================================================"
