#!/bin/bash
# ============================================================
# V10/V11/V12: Slim run — 핵심 configs only, GPU 1
# ============================================================
set -e

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COCO_PROMPTS="$BASE/prompts/coco_30.txt"
CONCEPT_DIR="$BASE/exemplars/sd14/concept_directions.pt"
GPU=1
OUTBASE="$BASE/outputs"

# Phase 0: concept directions
if [ ! -f "$CONCEPT_DIR" ]; then
    echo ">>> Phase 0: Preparing concept directions..."
    mkdir -p "$BASE/exemplars/sd14"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$BASE/prepare_concept_subspace.py" \
        --output "$CONCEPT_DIR" --steps 50 --cfg_scale 7.5 --seed 42 --batch_size 8 \
        2>&1 | tee "$BASE/exemplars/sd14/prepare.log"
fi

# ============================================================
# Core experiments (sequential on single GPU)
# ============================================================
run_experiment() {
    local name=$1 ver=$2 mode=$3
    shift 3
    local extras="$@"
    local script="$BASE/generate_${ver}.py"
    local outdir="$OUTBASE/$ver/$name"
    local prompt_file="$PROMPTS"

    if [[ "$extras" == *"--coco"* ]]; then
        prompt_file="$COCO_PROMPTS"
        extras="${extras//--coco/}"
    fi

    if [ -d "$outdir" ] && ls "$outdir"/*.png &>/dev/null 2>&1; then
        echo "  [SKIP] $name"
        return 0
    fi

    mkdir -p "$outdir"
    echo ">>> [$name] ($ver, $mode) started $(date '+%H:%M')"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$script" \
        --ckpt "CompVis/stable-diffusion-v1-4" \
        --prompts "$prompt_file" \
        --outdir "$outdir" \
        --nsamples 4 --steps 50 --seed 42 \
        --exemplar_mode exemplar \
        --concept_dir_path "$CONCEPT_DIR" \
        --cas_threshold 0.6 --spatial_threshold 0.3 \
        --sigmoid_alpha 10 --cas_sticky \
        --guide_mode $mode \
        $extras \
        2>&1 | tail -5
    echo "  DONE $(date '+%H:%M')"
}

echo "============================================================"
echo "Phase 1: Generation (5 core experiments on GPU $GPU)"
echo "============================================================"

# 1. v10 projection: surgical nudity removal
run_experiment "v10_proj_ts2_as1" v10 proj_anchor \
    --target_scale 2.0 --anchor_scale 1.0

# 2. v10 hybrid fidelity: v7 hybrid with deviation clamping (compare with v7_hyb_ts15_as15)
run_experiment "v10_hfid_ts15_as15_d03" v10 hybrid_fidelity \
    --target_scale 15 --anchor_scale 15 --max_deviation 0.3

# 3. v11 stochastic ensemble: K=4 diverse anchors + projection
run_experiment "v11_proj_K4_eta03" v11 proj_anchor \
    --target_scale 2.0 --anchor_scale 1.0 --K_ensemble 4 --eta 0.3 --ensemble_mode best

# 4. v12 cross-attention WHERE + projection HOW
run_experiment "v12_xattn_proj_ts2_as1" v12 proj_anchor \
    --spatial_mode crossattn --target_scale 2.0 --anchor_scale 1.0

# 5. v10 projection with adaptive CAS
run_experiment "v10_proj_ts3_as1_acas" v10 proj_anchor \
    --target_scale 3.0 --anchor_scale 1.0 --adaptive_cas

echo ""
echo "============================================================"
echo "Phase 2: NudeNet Evaluation"
echo "============================================================"

for ver in v10 v11 v12; do
    for d in "$OUTBASE/$ver"/*/; do
        name=$(basename "$d")
        if [ ! -f "$d/results_nudenet.txt" ] && ls "$d"/*.png &>/dev/null 2>&1; then
            echo "  [NN] $name"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON $EVAL_NN "$d" 2>/dev/null
        fi
    done
done

echo ""
echo "============================================================"
echo "Phase 3: Qwen3-VL Evaluation"
echo "============================================================"

for ver in v10 v11 v12; do
    for d in "$OUTBASE/$ver"/*/; do
        name=$(basename "$d")
        if [ ! -f "$d/categories_qwen3_vl_nudity.json" ] && \
           [ ! -f "$d/categories_qwen_nudity.json" ] && \
           ls "$d"/*.png &>/dev/null 2>&1; then
            echo "  [VLM] $name"
            CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON $EVAL_VLM "$d" nudity qwen 2>/dev/null
        fi
    done
done

echo ""
echo "============================================================"
echo "Phase 4: Results"
echo "============================================================"

$PYTHON << 'PYEOF'
import json, os, glob, re

base = "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"
print(f"{'Config':<40} {'NotRel':>6} {'Safe':>6} {'Part':>6} {'Full':>6} {'SR(%)':>7} {'NN_%':>8}")
print("-" * 85)

for ver in ["v7", "v10", "v11", "v12"]:
    ver_base = os.path.join(base, ver)
    if not os.path.isdir(ver_base):
        continue
    print(f"\n--- {ver.upper()} ---")
    for d in sorted(glob.glob(os.path.join(ver_base, "*/"))):
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
        print(f"  {name:<38} {nr:>6} {s:>6} {p:>6} {fl:>6} {sr:>7} {nn_pct:>8}")

PYEOF

echo ""
echo "ALL COMPLETE! $(date)"
