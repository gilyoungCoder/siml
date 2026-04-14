#!/bin/bash
set -e
PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
SAFREE="/mnt/home3/yhgil99/unlearning/unlearning-baselines/SAFREE_github"
SAVE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
cd $SAFREE

mkdir -p $SAVE/mma/safe $SAVE/mma/unsafe $SAVE/mma/all
mkdir -p $SAVE/unlearndiff/safe $SAVE/unlearndiff/unsafe $SAVE/unlearndiff/all

# MMA — need CSV format. Convert txt to csv first.
python3 -c "
lines = open('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mma.txt').readlines()
with open('/tmp/mma_safree.csv', 'w') as f:
    f.write('prompt\n')
    for l in lines:
        l = l.strip().replace('\"', '\"\"')
        if l: f.write(f'\"{l}\"\n')
print(f'Wrote {len([l for l in lines if l.strip()])} prompts')
"

echo "[GPU0] SAFREE MMA (1000 prompts)"
CUDA_VISIBLE_DEVICES=0 $PYTHON -s generate_safree.py \
    --config ./configs/sd_config.json \
    --data /tmp/mma_safree.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --num-samples 1 --erase-id std \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category nudity \
    --save-dir $SAVE/mma \
    --safree -svf -lra 2>&1 | tail -3 &

# UnlearnDiff — same treatment
python3 -c "
lines = open('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/unlearndiff.txt').readlines()
with open('/tmp/udiff_safree.csv', 'w') as f:
    f.write('prompt\n')
    for l in lines:
        l = l.strip().replace('\"', '\"\"')
        if l: f.write(f'\"{l}\"\n')
print(f'Wrote {len([l for l in lines if l.strip()])} prompts')
"

echo "[GPU1] SAFREE UnlearnDiff (142 prompts)"
CUDA_VISIBLE_DEVICES=1 $PYTHON -s generate_safree.py \
    --config ./configs/sd_config.json \
    --data /tmp/udiff_safree.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --num-samples 1 --erase-id std \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category nudity \
    --save-dir $SAVE/unlearndiff \
    --safree -svf -lra 2>&1 | tail -3 &

wait
echo "=== SAFREE MMA+UDiff COMPLETE $(date) ==="
