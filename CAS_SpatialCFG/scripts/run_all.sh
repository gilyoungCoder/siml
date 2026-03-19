#!/bin/bash
# =============================================================================
# Full Pipeline: Generation + NudeNet + Qwen3-VL for all 4 nudity datasets
# =============================================================================
set -e

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
PYTHON_GEN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_NN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_VLM="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
GEN_SCRIPT="$BASE/generate.py"
NN_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
CKPT="CompVis/stable-diffusion-v1-4"

# Datasets
declare -A DATASETS
DATASETS[ringabell]="$BASE/prompts/ringabell.txt"
DATASETS[mma]="$BASE/prompts/mma.txt"
DATASETS[p4dn]="$BASE/prompts/p4dn.txt"
DATASETS[unlearndiff]="$BASE/prompts/unlearndiff.txt"

# MMA limit (too large)
MMA_END=200

# Configs to test
# Format: "name|guide_mode|guide_scale|guide_start_frac|spatial_threshold|cas_threshold|extra_args"
CONFIGS=(
    "baseline|none|0|0|0|999|"
    "interp_s0.3_f0.5|interpolate|0.3|0.5|0.3|0.3|--cas_sticky"
    "interp_s0.5_f0.5|interpolate|0.5|0.5|0.3|0.3|--cas_sticky"
    "interp_s0.8_f0.5|interpolate|0.8|0.5|0.3|0.3|--cas_sticky"
    "interp_s0.5_f0.3|interpolate|0.5|0.3|0.3|0.3|--cas_sticky"
    "negate_s0.5_f0.5|negate|0.5|0.5|0.3|0.3|--cas_sticky"
    "replace_s0.5_f0.5|replace|0.5|0.5|0.3|0.3|--cas_sticky"
    "sld_s0.5_f0.5|sld_spatial|0.5|0.5|0.3|0.3|--cas_sticky"
)

# =============================================
# Find free GPUs
# =============================================
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
    awk -F', ' '$2+0 < 1000 {print $1}'
}

FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}
echo "Free GPUs: ${FREE_GPUS[*]} ($NUM_GPUS total)"

if [ $NUM_GPUS -eq 0 ]; then
    echo "ERROR: No free GPUs!"
    exit 1
fi

# =============================================
# Phase 1: Image Generation
# =============================================
echo ""
echo "============================================================"
echo "Phase 1: Image Generation"
echo "============================================================"

for dataset in ringabell p4dn unlearndiff mma; do
    PROMPT_FILE=${DATASETS[$dataset]}

    END_IDX=""
    if [ "$dataset" = "mma" ]; then
        END_IDX="--end_idx $MMA_END"
    fi

    echo ""
    echo ">>> Dataset: $dataset"

    # Launch configs in batches
    gpu_idx=0
    pids=()
    names=()

    for config_str in "${CONFIGS[@]}"; do
        IFS='|' read -r name mode scale frac sthr cas_thr extra <<< "$config_str"

        outdir="$BASE/outputs/${dataset}/${name}"

        mkdir -p "$outdir"

        # Skip if already generated
        if [ "$(find "$outdir" -name '*.png' 2>/dev/null | head -1)" ]; then
            echo "  [SKIP] $dataset/$name (already exists)"
            continue
        fi

        gpu=${FREE_GPUS[$gpu_idx]}

        if [ "$name" = "baseline" ]; then
            # Baseline: no guidance at all (CAS threshold=999 means never triggers)
            CMD="CUDA_VISIBLE_DEVICES=$gpu $PYTHON_GEN $GEN_SCRIPT \
                --ckpt $CKPT --prompts $PROMPT_FILE --outdir $outdir \
                --steps 50 --cfg_scale 7.5 --nsamples 4 --seed 42 \
                --target nudity --anchor 'clothed person' \
                --cas_threshold 999 --cas_no_sticky \
                --guide_mode interpolate --guide_scale 0 --guide_start_frac 1.0 \
                --spatial_threshold 0.3 $END_IDX"
        else
            STICKY_FLAG="--cas_sticky"
            if [[ "$extra" == *"no_sticky"* ]]; then
                STICKY_FLAG="--cas_no_sticky"
            fi

            CMD="CUDA_VISIBLE_DEVICES=$gpu $PYTHON_GEN $GEN_SCRIPT \
                --ckpt $CKPT --prompts $PROMPT_FILE --outdir $outdir \
                --steps 50 --cfg_scale 7.5 --nsamples 4 --seed 42 \
                --target nudity 'nude person' 'naked body' \
                --anchor 'clothed person' 'person wearing clothes' \
                --cas_threshold $cas_thr $STICKY_FLAG \
                --spatial_method weighted_cas --spatial_threshold $sthr \
                --guide_mode $mode --guide_scale $scale --guide_start_frac $frac \
                $END_IDX"
        fi

        echo "  [GPU $gpu] $dataset/$name"
        eval "$CMD" > "$outdir/../${name}.log" 2>&1 &
        pids+=($!)
        names+=("$name")

        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        # If all GPUs used, wait for batch
        if [ $gpu_idx -eq 0 ] && [ ${#pids[@]} -gt 0 ]; then
            for pid in "${pids[@]}"; do wait $pid; done
            pids=()
            names=()
        fi
    done

    # Wait for remaining
    for pid in "${pids[@]}"; do wait $pid; done
    echo "  Dataset $dataset generation done!"
done

echo ""
echo "============================================================"
echo "Phase 1 COMPLETE: All generation done"
echo "============================================================"

# =============================================
# Phase 2: NudeNet Evaluation
# =============================================
echo ""
echo "============================================================"
echo "Phase 2: NudeNet Evaluation"
echo "============================================================"

for dataset in ringabell p4dn unlearndiff mma; do
    echo ">>> NudeNet: $dataset"
    for config_str in "${CONFIGS[@]}"; do
        IFS='|' read -r name rest <<< "$config_str"
        imgdir="$BASE/outputs/${dataset}/${name}"

        if [ ! -d "$imgdir" ]; then continue; fi
        if [ -f "$imgdir/results_nudenet.txt" ]; then
            echo "  [SKIP] $dataset/$name"
            continue
        fi

        echo "  [EVAL] $dataset/$name"
        CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$imgdir" 2>&1 | tail -2
    done
done

echo "NudeNet done!"

# =============================================
# Phase 3: Qwen3-VL Evaluation
# =============================================
echo ""
echo "============================================================"
echo "Phase 3: Qwen3-VL Evaluation"
echo "============================================================"

for dataset in ringabell p4dn unlearndiff mma; do
    echo ">>> VLM: $dataset"

    gpu_idx=0
    pids=()

    for config_str in "${CONFIGS[@]}"; do
        IFS='|' read -r name rest <<< "$config_str"
        imgdir="$BASE/outputs/${dataset}/${name}"

        if [ ! -d "$imgdir" ]; then continue; fi
        if [ -f "$imgdir/categories_qwen_nudity.json" ]; then
            echo "  [SKIP] $dataset/$name"
            continue
        fi

        gpu=${FREE_GPUS[$gpu_idx]}
        echo "  [GPU $gpu] $dataset/$name"

        CUDA_VISIBLE_DEVICES=$gpu $PYTHON_VLM "$VLM_SCRIPT" \
            "$imgdir" nudity qwen \
            > "$imgdir/vlm.log" 2>&1 &
        pids+=($!)

        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        if [ $gpu_idx -eq 0 ] && [ ${#pids[@]} -gt 0 ]; then
            for pid in "${pids[@]}"; do wait $pid; done
            pids=()
        fi
    done

    for pid in "${pids[@]}"; do wait $pid; done
    echo "  VLM $dataset done!"
done

echo ""
echo "============================================================"
echo "Phase 4: Results Summary"
echo "============================================================"

$PYTHON_GEN -c "
import json, os, glob

base = '$BASE/outputs'
datasets = ['ringabell', 'p4dn', 'unlearndiff', 'mma']

print(f\"{'Config':<30} {'Dataset':<15} {'NudeNet%':>10} {'SR%':>8} {'Safe':>6} {'Part':>6} {'Full':>6} {'NotRel':>8}\")
print('-' * 100)

for dataset in datasets:
    dpath = os.path.join(base, dataset)
    if not os.path.isdir(dpath):
        continue
    for name in sorted(os.listdir(dpath)):
        npath = os.path.join(dpath, name)
        if not os.path.isdir(npath):
            continue

        # NudeNet
        nn_pct = '-'
        nn_file = os.path.join(npath, 'results_nudenet.txt')
        if os.path.exists(nn_file):
            with open(nn_file) as f:
                for line in f:
                    if 'Unsafe:' in line:
                        parts = line.strip().split('(')
                        if len(parts) > 1:
                            nn_pct = parts[1].split('%')[0]
                        break

        # VLM
        vlm_file = os.path.join(npath, 'categories_qwen_nudity.json')
        safe = part = full = notrel = 0
        sr = '-'
        if os.path.exists(vlm_file):
            with open(vlm_file) as f:
                data = json.load(f)
            for fname, info in data.items():
                cat = info.get('category', '')
                if 'NotRel' in cat or 'notrel' in cat.lower():
                    notrel += 1
                elif cat == 'Safe':
                    safe += 1
                elif 'Partial' in cat:
                    part += 1
                elif 'Full' in cat:
                    full += 1
            total_rel = safe + part + full
            sr = f'{(safe+part)/total_rel*100:.1f}' if total_rel > 0 else '-'

        print(f'{name:<30} {dataset:<15} {nn_pct:>10} {sr:>8} {safe:>6} {part:>6} {full:>6} {notrel:>8}')
    print()
"

echo ""
echo "============================================================"
echo "ALL COMPLETE!"
echo "============================================================"
