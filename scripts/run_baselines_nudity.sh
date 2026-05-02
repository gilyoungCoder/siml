#!/bin/bash
# ============================================================================
# Baseline Experiments: Safe_Denoiser, SGF, SDErasure × 4 Nudity Datasets
# GPU allocation: 1,2,3,5,6,7 on SIML01
# ============================================================================
set -e

BASE=/mnt/home3/yhgil99/unlearning
SD_DIR=$BASE/unlearning-baselines/Safe_Denoiser_official
SGF_DIR=$BASE/unlearning-baselines/SGF_official/nudity_sdv1
SDE_DIR=$BASE/SDErasure
VENV_PY=$SD_DIR/.venv/bin/python3.10
SAFREE_PY=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10

# NudeNet path
NUDENET=$SD_DIR/pretrained/classifier_model.onnx

# Output base
OUT_BASE=$BASE/unlearning-baselines/outputs

# Datasets
RAB_CSV=$SD_DIR/datasets/nudity-ring-a-bell.csv
MMA_CSV=$SD_DIR/datasets/mma-diffusion-nsfw-adv-prompts.csv
UDN_CSV=$SD_DIR/datasets/nudity.csv            # UnlearnDiff
P4DN_CSV=$SD_DIR/datasets/p4dn_16_prompt.csv
I2P_SEXUAL_CSV=$BASE/SAFREE/datasets/i2p_categories/i2p_sexual.csv

# ============================================================================
# Step 0: Generate negative reference images (if not already done)
# ============================================================================
generate_negative_images() {
    local GPU=$1
    local OUTPUT=$SD_DIR/datasets/nudity/i2p_sexual
    local COUNT=$(find $OUTPUT -name "*.png" 2>/dev/null | wc -l)

    if [ "$COUNT" -ge 500 ]; then
        echo "[Step 0] Negative images already exist ($COUNT images). Skipping."
        return
    fi

    echo "[Step 0] Generating negative reference images on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $SAFREE_PY $BASE/scripts/generate_negative_images.py \
        --csv $I2P_SEXUAL_CSV \
        --output_dir $OUTPUT \
        --device cuda:0 \
        --max_images 600 \
        --seed 42

    # Copy to SGF datasets too
    local SGF_OUTPUT=$SGF_DIR/datasets/nudity/i2p_sexual
    mkdir -p $SGF_OUTPUT
    cp $OUTPUT/*.png $SGF_OUTPUT/
    echo "[Step 0] Negative images copied to SGF datasets."
}

# ============================================================================
# Safe_Denoiser experiments
# ============================================================================
run_safe_denoiser() {
    local GPU=$1
    local DATA_CSV=$2
    local DATASET_NAME=$3
    local SAVE_DIR=$OUT_BASE/safe_denoiser/$DATASET_NAME

    echo "[Safe_Denoiser] $DATASET_NAME on GPU $GPU -> $SAVE_DIR"
    mkdir -p $SAVE_DIR

    cd $SD_DIR
    CUDA_VISIBLE_DEVICES=$GPU $VENV_PY run_nudity.py \
        --config configs/base/vanilla/safree_neg_prompt_config.json \
        --data $DATA_CSV \
        --save-dir $SAVE_DIR \
        --erase_id safree_neg_prompt_rep_threshold_time \
        --category nudity \
        --task_config configs/nudity/safe_denoiser.yaml \
        --safe_level MEDIUM \
        --device cuda:0 \
        --nudenet-path $NUDENET \
        --nudity_thr 0.6 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        2>&1 | tee $SAVE_DIR/run.log
}

# ============================================================================
# SGF experiments
# ============================================================================
run_sgf() {
    local GPU=$1
    local DATA_CSV=$2
    local DATASET_NAME=$3
    local SAVE_DIR=$OUT_BASE/sgf/$DATASET_NAME

    echo "[SGF] $DATASET_NAME on GPU $GPU -> $SAVE_DIR"
    mkdir -p $SAVE_DIR

    cd $SGF_DIR
    CUDA_VISIBLE_DEVICES=$GPU $VENV_PY generate_unsafe_sgf.py \
        --config configs/base/safree_neg_prompt_config.json \
        --data $DATA_CSV \
        --save-dir $SAVE_DIR \
        --erase_id safree_neg_prompt_rep_time \
        --category nudity \
        --task_config configs/sgf/sgf.yaml \
        --safe_level MEDIUM \
        --device cuda:0 \
        --nudenet-path $NUDENET \
        --nudity_thr 0.6 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        2>&1 | tee $SAVE_DIR/run.log
}

# ============================================================================
# SDErasure experiments (use existing trained UNet)
# ============================================================================
run_sderasure() {
    local GPU=$1
    local DATA_CSV=$2
    local DATASET_NAME=$3
    local UNET_DIR=$SDE_DIR/outputs/sderasure_nudity/unet
    local SAVE_DIR=$OUT_BASE/sderasure_nudity/$DATASET_NAME

    if [ ! -d "$UNET_DIR" ]; then
        echo "[SDErasure] ERROR: Trained UNet not found at $UNET_DIR"
        return 1
    fi

    echo "[SDErasure] $DATASET_NAME on GPU $GPU -> $SAVE_DIR"
    mkdir -p $SAVE_DIR

    CUDA_VISIBLE_DEVICES=$GPU $SAFREE_PY $BASE/scripts/generate_sderasure_from_csv.py \
        --model_id CompVis/stable-diffusion-v1-4 \
        --unet_dir $UNET_DIR \
        --csv $DATA_CSV \
        --output_dir $SAVE_DIR \
        --device cuda:0 \
        --num_inference_steps 50 \
        --guidance_scale 7.5 \
        --seed 42 \
        2>&1 | tee $SAVE_DIR/run.log
}

# ============================================================================
# Main: dispatch all experiments
# ============================================================================
main() {
    echo "============================================"
    echo "Baseline Nudity Experiments"
    echo "Server: $(hostname), GPUs: 3-7"
    echo "Date: $(date)"
    echo "============================================"

    # Step 0: Generate negative images first (blocking)
    generate_negative_images 1

    # Copy dataset CSVs to SGF dir (in case not done)
    for f in nudity-ring-a-bell.csv mma-diffusion-nsfw-adv-prompts.csv nudity.csv p4dn_16_prompt.csv; do
        cp -n $SD_DIR/datasets/$f $SGF_DIR/datasets/ 2>/dev/null || true
    done

    echo ""
    echo "Starting experiments in parallel..."
    echo ""

    # GPU 1: Safe_Denoiser Ring-A-Bell + P4DN
    (run_safe_denoiser 1 $RAB_CSV ringabell && run_safe_denoiser 1 $P4DN_CSV p4dn) &
    PID_SD1=$!

    # GPU 2: Safe_Denoiser MMA + UnlearnDiff
    (run_safe_denoiser 2 $MMA_CSV mma && run_safe_denoiser 2 $UDN_CSV unlearndiff) &
    PID_SD2=$!

    # GPU 3: SGF Ring-A-Bell + P4DN
    (run_sgf 3 $RAB_CSV ringabell && run_sgf 3 $P4DN_CSV p4dn) &
    PID_SGF1=$!

    # GPU 5: SGF MMA + UnlearnDiff
    (run_sgf 5 $MMA_CSV mma && run_sgf 5 $UDN_CSV unlearndiff) &
    PID_SGF2=$!

    # GPU 6: SDErasure Ring-A-Bell + MMA
    (run_sderasure 6 $RAB_CSV ringabell && run_sderasure 6 $MMA_CSV mma) &
    PID_SDE1=$!

    # GPU 7: SDErasure UnlearnDiff + P4DN
    (run_sderasure 7 $UDN_CSV unlearndiff && run_sderasure 7 $P4DN_CSV p4dn) &
    PID_SDE2=$!

    echo "PIDs: SD1=$PID_SD1 SD2=$PID_SD2 SGF1=$PID_SGF1 SGF2=$PID_SGF2 SDE1=$PID_SDE1 SDE2=$PID_SDE2"

    # Wait for all
    wait $PID_SD1 && echo "Safe_Denoiser GPU1 done" || echo "Safe_Denoiser GPU1 FAILED"
    wait $PID_SD2 && echo "Safe_Denoiser GPU2 done" || echo "Safe_Denoiser GPU2 FAILED"
    wait $PID_SGF1 && echo "SGF GPU3 done" || echo "SGF GPU3 FAILED"
    wait $PID_SGF2 && echo "SGF GPU5 done" || echo "SGF GPU5 FAILED"
    wait $PID_SDE1 && echo "SDErasure GPU6 done" || echo "SDErasure GPU6 FAILED"
    wait $PID_SDE2 && echo "SDErasure GPU7 done" || echo "SDErasure GPU7 FAILED"

    echo ""
    echo "============================================"
    echo "All experiments completed: $(date)"
    echo "============================================"
}

# Allow running individual methods
case "${1:-all}" in
    neg_images)   generate_negative_images ${2:-3} ;;
    safe_denoiser) run_safe_denoiser ${2:-3} ${3:-$RAB_CSV} ${4:-ringabell} ;;
    sgf)          run_sgf ${2:-5} ${3:-$RAB_CSV} ${4:-ringabell} ;;
    sderasure)    run_sderasure ${2:-7} ${3:-$RAB_CSV} ${4:-ringabell} ;;
    all)          main ;;
    *)            echo "Usage: $0 [all|neg_images|safe_denoiser|sgf|sderasure] [GPU] [CSV] [NAME]" ;;
esac
