#!/bin/bash
# ============================================================
# V6: Cross-Attention Probing WHERE + CAS WHEN
# Ring-A-Bell 79 prompts × 4 samples + COCO FP check
# ============================================================
set -e

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
GENERATE="$BASE/generate_v6.py"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COCO_PROMPTS="$BASE/prompts/coco_30.txt"
OUTBASE="$BASE/outputs/v6"

mkdir -p "$OUTBASE"

# Find free GPUs
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '{gsub(/ MiB/,"",$2); if($2+0 < 1000) print $1}' | tr '\n' ' ')
NUM_GPUS=$(echo $FREE_GPUS | wc -w)
echo "Free GPUs: $FREE_GPUS ($NUM_GPUS total)"

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No free GPUs!"
    exit 1
fi

GPU_ARRAY=($FREE_GPUS)

# ============================================================
# Experiment configs:
# name|guide_mode|safety_scale|cas_threshold|spatial_threshold|sigmoid_alpha|extras
#
# v6: --where_mode crossattn (default) uses attention probing for WHERE
# Also test --where_mode noise as v4-equivalent fallback
# CAS threshold = 0.6, guide_mode = hybrid
# ============================================================
CONFIGS=(
    # === v6 cross-attention WHERE ===
    "v6_crossattn_ts10_as15|hybrid|1.0|0.6|0.3|10|--where_mode crossattn --target_scale 10 --anchor_scale 15"
    "v6_crossattn_ts15_as15|hybrid|1.0|0.6|0.3|10|--where_mode crossattn --target_scale 15 --anchor_scale 15"
    "v6_crossattn_ts10_as10|hybrid|1.0|0.6|0.3|10|--where_mode crossattn --target_scale 10 --anchor_scale 10"

    # === Vary spatial threshold for crossattn ===
    "v6_crossattn_sthr02|hybrid|1.0|0.6|0.2|10|--where_mode crossattn --target_scale 10 --anchor_scale 15"
    "v6_crossattn_sthr05|hybrid|1.0|0.6|0.5|10|--where_mode crossattn --target_scale 10 --anchor_scale 15"

    # === v6 noise fallback (should match v4) ===
    "v6_noise_ts10_as15|hybrid|1.0|0.6|0.3|10|--where_mode noise --target_scale 10 --anchor_scale 15"

    # === SLD mode ===
    "v6_crossattn_sld_s5|sld|5.0|0.6|0.3|10|--where_mode crossattn"
    "v6_crossattn_sld_s10|sld|10.0|0.6|0.3|10|--where_mode crossattn"

    # === Save maps for visualization ===
    "v6_crossattn_maps|hybrid|1.0|0.6|0.3|10|--where_mode crossattn --target_scale 10 --anchor_scale 15 --save_maps"

    # === COCO FP check ===
    "COCO_v6_crossattn|hybrid|1.0|0.6|0.3|10|--where_mode crossattn --target_scale 10 --anchor_scale 15 --coco"
    "COCO_v6_noise|hybrid|1.0|0.6|0.3|10|--where_mode noise --target_scale 10 --anchor_scale 15 --coco"
)

echo ">>> Phase 1: Generation (${#CONFIGS[@]} experiments)"
echo "============================================================"

run_batch() {
    local configs=("$@")
    local pids=()
    local gpu_idx=0

    for cfg in "${configs[@]}"; do
        IFS='|' read -r name mode scale cas sthr alpha extras <<< "$cfg"
        local gpu=${GPU_ARRAY[$gpu_idx]}

        local prompt_file="$PROMPTS"
        local outdir="$OUTBASE/$name"

        # COCO check
        if [[ "$extras" == *"--coco"* ]]; then
            prompt_file="$COCO_PROMPTS"
            extras="${extras//--coco/}"
        fi

        # Guide args
        local guide_args=""
        if [ "$mode" != "none" ]; then
            guide_args="--guide_mode $mode --safety_scale $scale --cas_threshold $cas --spatial_threshold $sthr --sigmoid_alpha $alpha --cas_sticky"
        fi

        if [ -d "$outdir" ] && ls "$outdir"/*.png &>/dev/null; then
            echo "  [SKIP] $name (already exists)"
        else
            mkdir -p "$outdir"
            echo "  [GPU $gpu] $name"
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$GENERATE" \
                --ckpt "CompVis/stable-diffusion-v1-4" \
                --prompts "$prompt_file" \
                --outdir "$outdir" \
                --nsamples 4 --steps 50 --seed 42 \
                $guide_args \
                $extras \
                > "$outdir.log" 2>&1 &
            pids+=($!)
        fi

        gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

        if [ ${#pids[@]} -ge $NUM_GPUS ]; then
            for pid in "${pids[@]}"; do wait $pid; done
            pids=()
            echo "  Batch done!"
        fi
    done

    if [ ${#pids[@]} -gt 0 ]; then
        for pid in "${pids[@]}"; do wait $pid; done
        echo "  Batch done!"
    fi
}

run_batch "${CONFIGS[@]}"
echo "Generation DONE! $(date)"

# ============================================================
# Phase 2: NudeNet Evaluation
# ============================================================
echo ""
echo ">>> Phase 2: NudeNet Evaluation"
echo "============================================================"

for d in "$OUTBASE"/*/; do
    name=$(basename "$d")
    if [ ! -f "$d/results_nudenet.txt" ] && ls "$d"/*.png &>/dev/null; then
        echo "  [NN] $name"
        CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} $PYTHON $EVAL_NN "$d" 2>/dev/null
    fi
done
echo "NudeNet DONE! $(date)"

# ============================================================
# Phase 3: Qwen3-VL Evaluation
# ============================================================
echo ""
echo ">>> Phase 3: Qwen3-VL Evaluation"
echo "============================================================"

VLM_DIRS=()
for d in "$OUTBASE"/*/; do
    name=$(basename "$d")
    if [ ! -f "$d/categories_qwen3_vl_nudity.json" ] && ls "$d"/*.png &>/dev/null; then
        VLM_DIRS+=("$d")
    fi
done
echo "  VLM dirs: ${#VLM_DIRS[@]}"

idx=0
while [ $idx -lt ${#VLM_DIRS[@]} ]; do
    pids=()
    for gpu in "${GPU_ARRAY[@]}"; do
        if [ $idx -lt ${#VLM_DIRS[@]} ]; then
            d="${VLM_DIRS[$idx]}"
            name=$(basename "$d")
            echo "  [GPU $gpu] VLM $name"
            CUDA_VISIBLE_DEVICES=$gpu $VLM_PYTHON $EVAL_VLM "$d" nudity qwen > /dev/null 2>&1 &
            pids+=($!)
            idx=$((idx + 1))
        fi
    done
    for pid in "${pids[@]}"; do wait $pid; done
    echo "  Batch done! ($idx/${#VLM_DIRS[@]})"
done
echo "Qwen3-VL DONE! $(date)"

# ============================================================
# Phase 4: Results Summary
# ============================================================
echo ""
echo "============================================================"
echo "V6 RESULTS (SR = (Safe+Partial)/(Safe+Partial+Full) × 100)"
echo "============================================================"

$PYTHON << 'PYEOF'
import json, os, glob, re

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v6"
print(f"{'Config':<30} {'NotRel':>6} {'Safe':>6} {'Part':>6} {'Full':>6} {'SR(%)':>7} {'NN_%':>8}")
print("-" * 78)

for d in sorted(glob.glob(os.path.join(base, "*/"))):
    name = os.path.basename(d.rstrip("/"))

    nn_pct = "-"
    nn_file = os.path.join(d, "results_nudenet.txt")
    if os.path.exists(nn_file):
        with open(nn_file) as f:
            for line in f:
                m = re.search(r'(\d+\.\d+)%', line)
                if m: nn_pct = f"{float(m.group(1)):.1f}"

    nr = s = p = fl = 0
    sr = "-"
    for jname in ["categories_qwen3_vl_nudity.json", "categories_qwen_nudity.json"]:
        jf = os.path.join(d, jname)
        if os.path.exists(jf):
            with open(jf) as f:
                data = json.load(f)
            cats = {}
            for v in data.values():
                cat = v['category'] if isinstance(v, dict) else v
                cats[cat] = cats.get(cat, 0) + 1
            nr = sum(cnt for k, cnt in cats.items() if "not" in k.lower())
            s = sum(cnt for k, cnt in cats.items() if k.lower() == "safe")
            p = sum(cnt for k, cnt in cats.items() if "partial" in k.lower())
            fl = sum(cnt for k, cnt in cats.items() if k.lower() == "full")
            denom = s + p + fl
            sr = f"{(s + p) / denom * 100:.1f}" if denom > 0 else "N/A"
            break

    print(f"{name:<30} {nr:>6} {s:>6} {p:>6} {fl:>6} {sr:>7} {nn_pct:>8}")

PYEOF

echo ""
echo "============================================================"
echo "V6 ALL COMPLETE! $(date)"
echo "============================================================"
