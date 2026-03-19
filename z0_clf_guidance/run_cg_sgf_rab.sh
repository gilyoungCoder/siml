#!/bin/bash
# Run spatial CG with SGF-style time window on Ring-A-Bell data
# Produces: SD1.4+CG(window), SD1.4+CG(full), for comparison
# Then evaluates with NudeNet + VLM

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/mnt/home/yhgil99/.conda/envs/sdd/bin/python"
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"

# Paths
CLASSIFIER_3CLASS="work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
CLASSIFIER_4CLASS="work_dirs/z0_resnet18_4class/checkpoint/step_18900/classifier.pth"
HARMFUL_STATS="harmful_stats.pt"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/datasets/nudity-ring-a-bell.csv"
EVAL_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
NUDENET_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/pretrained/classifier_model.onnx"
NUDENET_EVAL="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/eval_nudenet_asr_tr.py"

MODEL="CompVis/stable-diffusion-v1-4"

# Common args
COMMON="--num_inference_steps 50 --cfg_scale 7.5 --seed 42 --nsamples 1 --architecture resnet18"

echo "=========================================="
echo "Spatial CG + SGF Time Window Experiments"
echo "=========================================="

# Experiment 1: SD1.4 + CG (SGF window: t=[800, 1000])
echo ""
echo ">>> [1/4] SD1.4 + CG (window t=[800,1000], scale=10)"
CUDA_VISIBLE_DEVICES=${1:-0} PYTHONNOUSERSITE=1 $PYTHON generate_sgf_window.py $MODEL \
    --prompt_file "$PROMPT_FILE" \
    --csv_prompt_column "sensitive prompt" \
    --classifier_ckpt "$CLASSIFIER_3CLASS" \
    --num_classes 3 --target_class 1 \
    --guidance_scale 10.0 \
    --guidance_start_t 1000 --guidance_end_t 800 \
    --spatial_mode gradcam \
    --spatial_threshold 0.3 \
    --gradcam_layer layer2 \
    --harmful_stats_path "$HARMFUL_STATS" \
    --output_dir output_img/cg_window_rab \
    $COMMON 2>&1 | tee logs/cg_window_rab.log

# Experiment 2: SD1.4 + CG (full: t=[0, 1000])
echo ""
echo ">>> [2/4] SD1.4 + CG (full, scale=10)"
CUDA_VISIBLE_DEVICES=${1:-0} PYTHONNOUSERSITE=1 $PYTHON generate_sgf_window.py $MODEL \
    --prompt_file "$PROMPT_FILE" \
    --csv_prompt_column "sensitive prompt" \
    --classifier_ckpt "$CLASSIFIER_3CLASS" \
    --num_classes 3 --target_class 1 \
    --guidance_scale 10.0 \
    --guidance_start_t 1000 --guidance_end_t 0 \
    --spatial_mode gradcam \
    --spatial_threshold 0.3 \
    --gradcam_layer layer2 \
    --harmful_stats_path "$HARMFUL_STATS" \
    --output_dir output_img/cg_full_rab \
    $COMMON 2>&1 | tee logs/cg_full_rab.log

echo ""
echo "All generation complete! Run eval separately."
