#!/bin/bash
# Launch z0 generation grid search on 8 GPUs
# RAB test + COCO

export PYTHONNOUSERSITE=1
PYTHON=/mnt/home/yhgil99/.conda/envs/safree/bin/python
cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

CLASSIFIER=./work_dirs/z0_resnet18_4class_ringabell/checkpoint/step_15900/classifier.pth
STATS=./harmful_stats_4class_ringabell_layer2.pt
RAB_TEST=/mnt/home/yhgil99/unlearning/rab_grid_search/data/ringabell_test.txt
COCO=/mnt/home/yhgil99/unlearning/prompts/coco/coco_10k.txt
OUT_BASE=/mnt/home/yhgil99/unlearning/rab_grid_search/results

COMMON_ARGS=(
    --classifier_ckpt $CLASSIFIER
    --harmful_stats_path $STATS
    --num_classes 4
    --guidance_mode safe_minus_harm
    --safe_classes 0 1
    --harm_classes 2 3
    --target_class 1
    --gradcam_layer layer2
    --num_gpus 8
    --gpu_ids 0,1,2,3,4,5,6,7
    --guidance_scales 5 10 20 50
    --spatial_modes none gradcam
    --spatial_thresholds 0.2 0.3 0.5 0.7
    --spatial_soft_options 0 1
    --threshold_schedules constant
    --harm_ratios 0.5 1.0 2.0
)

echo "=========================================="
echo "Phase 2a: Z0 Grid Search on RAB test"
echo "=========================================="
$PYTHON grid_search_spatial_cg.py \
    --prompt_file $RAB_TEST \
    --output_root $OUT_BASE/z0_gen_rab_test \
    "${COMMON_ARGS[@]}"

echo ""
echo "=========================================="
echo "Phase 2b: Z0 Grid Search on COCO (50 prompts)"
echo "=========================================="

# Create COCO subset
head -50 $COCO > /tmp/coco_50.txt

$PYTHON grid_search_spatial_cg.py \
    --prompt_file /tmp/coco_50.txt \
    --output_root $OUT_BASE/z0_gen_coco50 \
    "${COMMON_ARGS[@]}"

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
