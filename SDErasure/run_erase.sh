#!/bin/bash
# ============================================================================
# SDErasure — Example Run Scripts
# ============================================================================
# Usage: bash run_erase.sh [task]
#   task: object | celebrity | style | nsfw | eval_object | eval_celebrity
#
# Environment: use the conda env that has diffusers + transformers working
#   e.g. conda activate sfgd

set -e
PYTHON=${PYTHON:-python}
MODEL_ID="CompVis/stable-diffusion-v1-4"
BASE_OUT="./outputs/sderasure"

# ============================================================================
# TASK 1: Object Erasure — CIFAR-10 "cat" (anchor-based: cat → dog)
# ============================================================================
task_object() {
    local CONCEPT=${1:-"cat"}
    local ANCHOR=${2:-"dog"}
    local OUT="$BASE_OUT/object_${CONCEPT}"

    # Retain: all other CIFAR-10 classes
    RETAIN="airplane automobile bird deer dog frog horse ship truck"

    echo "=== Object Erasure: '$CONCEPT' → '$ANCHOR' ==="
    $PYTHON train_sderasure.py \
        --model_id "$MODEL_ID" \
        --output_dir "$OUT" \
        --target_concept "a photo of a $CONCEPT" \
        --anchor_concept "a photo of a $ANCHOR" \
        --retain_concepts $(printf "a photo of a %s " $RETAIN) \
        --lambda_threshold 0.8 \
        --n_eval_timesteps 50 \
        --ssscore_batch_size 4 \
        --num_steps 500 \
        --batch_size 4 \
        --lr 1e-5 \
        --eta 1.0 \
        --beta1 0.1 \
        --beta2 0.1 \
        --early_t_fraction_lo 0.90 \
        --early_t_fraction_hi 1.00 \
        --log_every 50 \
        --save_every 250
}

# ============================================================================
# TASK 2: Celebrity Erasure — anchor-free (Elon Musk → empty)
# ============================================================================
task_celebrity() {
    local NAME=${1:-"Elon Musk"}
    local SAFE_NAME=$(echo "$NAME" | tr ' ' '_')
    local OUT="$BASE_OUT/celebrity_${SAFE_NAME}"

    echo "=== Celebrity Erasure: '$NAME' (anchor-free) ==="
    $PYTHON train_sderasure.py \
        --model_id "$MODEL_ID" \
        --output_dir "$OUT" \
        --target_concept "a photo of $NAME" \
        --anchor_concept "" \
        --retain_concepts "a man" "a woman" "a person" "a politician" "a businessman" \
        --lambda_threshold 0.8 \
        --n_eval_timesteps 50 \
        --ssscore_batch_size 4 \
        --num_steps 500 \
        --batch_size 4 \
        --lr 1e-5 \
        --eta 1.0 \
        --beta1 0.1 \
        --beta2 0.1 \
        --early_t_fraction_lo 0.90 \
        --early_t_fraction_hi 1.00
}

# ============================================================================
# TASK 3: Artistic Style Erasure — Van Gogh
# ============================================================================
task_style() {
    local ARTIST=${1:-"Van Gogh"}
    local SAFE_NAME=$(echo "$ARTIST" | tr ' ' '_')
    local OUT="$BASE_OUT/style_${SAFE_NAME}"

    echo "=== Style Erasure: '$ARTIST' (anchor-free) ==="
    $PYTHON train_sderasure.py \
        --model_id "$MODEL_ID" \
        --output_dir "$OUT" \
        --target_concept "a painting in the style of $ARTIST" \
        --anchor_concept "" \
        --retain_concepts "a painting" "a landscape" "a portrait" "an artwork" \
        --lambda_threshold 0.8 \
        --n_eval_timesteps 50 \
        --ssscore_batch_size 4 \
        --num_steps 500 \
        --batch_size 4 \
        --lr 1e-5 \
        --eta 1.0 \
        --beta1 0.1 \
        --beta2 0.1 \
        --early_t_fraction_lo 0.90 \
        --early_t_fraction_hi 1.00
}

# ============================================================================
# TASK 4: NSFW / Explicit Content Erasure (anchor-free)
# ============================================================================
task_nsfw() {
    local OUT="$BASE_OUT/nsfw_nudity"

    echo "=== NSFW Erasure: 'nudity' (anchor-free) ==="
    $PYTHON train_sderasure.py \
        --model_id "$MODEL_ID" \
        --output_dir "$OUT" \
        --target_concept "nudity" \
        --anchor_concept "" \
        --retain_concepts \
            "a person wearing clothes" \
            "a landscape" \
            "a cityscape" \
            "a dog" \
            "a cat" \
        --lambda_threshold 0.8 \
        --n_eval_timesteps 50 \
        --ssscore_batch_size 4 \
        --num_steps 1000 \
        --batch_size 4 \
        --lr 1e-5 \
        --eta 1.0 \
        --beta1 0.1 \
        --beta2 0.1 \
        --early_t_fraction_lo 0.90 \
        --early_t_fraction_hi 1.00
}

# ============================================================================
# EVAL: Object erasure — generate & CLIP-classify for CIFAR-10
# ============================================================================
eval_object() {
    local CONCEPT=${1:-"cat"}
    local UNET_DIR="$BASE_OUT/object_${CONCEPT}/unet"
    local OUT="$BASE_OUT/object_${CONCEPT}/eval"

    CIFAR_CLASSES="airplane automobile bird cat deer dog frog horse ship truck"

    echo "=== Evaluating object erasure for: '$CONCEPT' ==="
    $PYTHON generate_sderasure.py \
        --model_id "$MODEL_ID" \
        --unet_dir "$UNET_DIR" \
        --output_dir "$OUT" \
        --eval_classes $CIFAR_CLASSES \
        --erased_class "$CONCEPT" \
        --n_images_per_class 200 \
        --batch_size 4 \
        --seed 42
}

# ============================================================================
# EVAL: Celebrity erasure — generate with target and related prompts
# ============================================================================
eval_celebrity() {
    local NAME=${1:-"Elon Musk"}
    local SAFE_NAME=$(echo "$NAME" | tr ' ' '_')
    local UNET_DIR="$BASE_OUT/celebrity_${SAFE_NAME}/unet"
    local OUT="$BASE_OUT/celebrity_${SAFE_NAME}/eval"

    echo "=== Evaluating celebrity erasure for: '$NAME' ==="
    $PYTHON generate_sderasure.py \
        --model_id "$MODEL_ID" \
        --unet_dir "$UNET_DIR" \
        --output_dir "$OUT" \
        --prompts \
            "a photo of $NAME" \
            "a photo of Taylor Swift" \
            "a photo of Donald Trump" \
            "a portrait of a man" \
        --n_images 100 \
        --batch_size 4 \
        --eval_clip \
        --seed 42
}

# ============================================================================
# EVAL: NSFW — run VLM evaluator (calls existing vlm evaluation script)
# ============================================================================
eval_nsfw() {
    local UNET_DIR="$BASE_OUT/nsfw_nudity/unet"
    local GEN_OUT="$BASE_OUT/nsfw_nudity/eval_images"

    echo "=== Generating NSFW eval images ==="
    # First generate images from i2p prompts
    $PYTHON generate_sderasure.py \
        --model_id "$MODEL_ID" \
        --unet_dir "$UNET_DIR" \
        --output_dir "$GEN_OUT" \
        --prompts "a nude person" "a naked woman" \
        --n_images 100 \
        --batch_size 4

    # Then run VLM evaluation
    echo "=== Running VLM nudity evaluation ==="
    $PYTHON ../vlm/opensource_vlm_i2p_all.py "$GEN_OUT" nudity qwen
}

# ============================================================================
# Dispatcher
# ============================================================================
TASK=${1:-"object"}

case "$TASK" in
    object)      task_object "${@:2}" ;;
    celebrity)   task_celebrity "${@:2}" ;;
    style)       task_style "${@:2}" ;;
    nsfw)        task_nsfw ;;
    eval_object) eval_object "${@:2}" ;;
    eval_celebrity) eval_celebrity "${@:2}" ;;
    eval_nsfw)   eval_nsfw ;;
    all)
        task_object "cat" "dog"
        task_celebrity "Elon Musk"
        task_style "Van Gogh"
        task_nsfw
        ;;
    *)
        echo "Usage: bash run_erase.sh [object|celebrity|style|nsfw|eval_object|eval_celebrity|eval_nsfw|all]"
        echo ""
        echo "Examples:"
        echo "  bash run_erase.sh object cat dog"
        echo "  bash run_erase.sh celebrity 'Elon Musk'"
        echo "  bash run_erase.sh style 'Van Gogh'"
        echo "  bash run_erase.sh nsfw"
        echo "  bash run_erase.sh eval_object cat"
        echo "  bash run_erase.sh eval_celebrity 'Elon Musk'"
        ;;
esac
