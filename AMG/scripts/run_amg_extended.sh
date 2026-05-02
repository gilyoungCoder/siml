#!/bin/bash
# ============================================================
# AMG Extended Grid Search
# More configs than initial run
# ============================================================
set -e

PYTHON="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
GENERATE="/mnt/home/yhgil99/unlearning/AMG/generate.py"
EVAL_NN="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
OUTBASE="/mnt/home/yhgil99/unlearning/AMG/outputs"
mkdir -p "$OUTBASE"

FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '{gsub(/ MiB/,"",$2); if($2+0 < 1000) print $1}' | tr '\n' ' ')
NUM_GPUS=$(echo $FREE_GPUS | wc -w)
echo "Free GPUs: $FREE_GPUS ($NUM_GPUS total)"
GPU_ARRAY=($FREE_GPUS)
if [ "$NUM_GPUS" -lt 1 ]; then echo "No free GPUs!"; exit 1; fi

# Extended configs - format: name|mode|scale|det_threshold|extras
CONFIGS=(
    # Scale sweep for SLD
    "sld_s2_d03|sld|2.0|0.3|"
    "sld_s4_d03|sld|4.0|0.3|"
    "sld_s6_d03|sld|6.0|0.3|"
    "sld_s8_d03|sld|8.0|0.3|"
    "sld_s10_d03|sld|10.0|0.3|"
    # Detection threshold sweep
    "sld_s5_d01|sld|5.0|0.1|"
    "sld_s5_d02|sld|5.0|0.2|"
    "sld_s5_d04|sld|5.0|0.4|"
    "sld_s5_d06|sld|5.0|0.6|"
    # Anchor shift scale sweep
    "ashift_s1_d03|sld|1.0|0.3|--anchor"
    "ashift_s2_d03|sld|2.0|0.3|--anchor"
    "ashift_s4_d03|sld|4.0|0.3|--anchor"
    "ashift_s7_d03|sld|7.0|0.3|--anchor"
    "ashift_s10_d03|sld|10.0|0.3|--anchor"
    # COCO FP
    "COCO_sld_s5|sld|5.0|0.3|--coco"
    "COCO_ashift_s5|sld|5.0|0.3|--anchor --coco"
)

echo "Total: ${#CONFIGS[@]} configs"

# Check which already exist
NEW_CONFIGS=()
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name rest <<< "$cfg"
    outdir="$OUTBASE/$name"
    if [ -d "$outdir" ] && [ "$(ls "$outdir"/*.png 2>/dev/null | wc -l)" -ge 100 ]; then
        echo "  [SKIP] $name"
    else
        NEW_CONFIGS+=("$cfg")
    fi
done

echo "New configs to run: ${#NEW_CONFIGS[@]}"

# Generation
idx=0
while [ $idx -lt ${#NEW_CONFIGS[@]} ]; do
    pids=()
    gpu_idx=0
    while [ $gpu_idx -lt $NUM_GPUS ] && [ $idx -lt ${#NEW_CONFIGS[@]} ]; do
        cfg="${NEW_CONFIGS[$idx]}"
        IFS='|' read -r name mode scale det extras <<< "$cfg"
        gpu=${GPU_ARRAY[$gpu_idx]}
        outdir="$OUTBASE/$name"
        mkdir -p "$outdir"

        prompt_file="$PROMPTS"
        [[ "$extras" == *"--coco"* ]] && prompt_file="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/prompts/coco_30.txt"

        echo "  [GPU $gpu] $name"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON $GENERATE \
            --ckpt "CompVis/stable-diffusion-v1-4" \
            --prompts "$prompt_file" \
            --outdir "$outdir" \
            --nsamples 4 --steps 50 --seed 42 \
            --guide_mode "$mode" --safety_scale "$scale" \
            --det_threshold "$det" --cas_sticky \
            > "$outdir.log" 2>&1 &
        pids+=($!)
        gpu_idx=$((gpu_idx + 1))
        idx=$((idx + 1))
    done
    if [ ${#pids[@]} -gt 0 ]; then
        for pid in "${pids[@]}"; do wait $pid; done
        echo "  Batch done!"
    fi
done

echo "Generation done! $(date)"

# NudeNet
for d in "$OUTBASE"/*/; do
    if [ ! -f "$d/results_nudenet.txt" ] && ls "$d"/*.png &>/dev/null; then
        CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} $PYTHON $EVAL_NN "$d" 2>/dev/null
    fi
done

# VLM
VLM_DIRS=()
for d in "$OUTBASE"/*/; do
    if [ ! -f "$d/categories_qwen3_vl_nudity.json" ] && ls "$d"/*.png &>/dev/null; then
        VLM_DIRS+=("$d")
    fi
done
idx=0
while [ $idx -lt ${#VLM_DIRS[@]} ]; do
    pids=()
    for gpu in "${GPU_ARRAY[@]}"; do
        if [ $idx -lt ${#VLM_DIRS[@]} ]; then
            CUDA_VISIBLE_DEVICES=$gpu $VLM_PYTHON $EVAL_VLM "${VLM_DIRS[$idx]}" nudity qwen > /dev/null 2>&1 &
            pids+=($!)
            idx=$((idx + 1))
        fi
    done
    for pid in "${pids[@]}"; do wait $pid; done
done

echo "ALL COMPLETE! $(date)"
