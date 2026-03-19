#!/bin/bash
# ============================================================
# SHGD Full Experiment Suite — 8 GPU parallel
# ============================================================
# Phase 1: Ablation (7 experiments → 7 GPUs parallel)
# Phase 2: Grid Search (quick grid → 8 GPUs parallel)
# Phase 3: COCO quality eval (best config → 8 GPUs parallel)
# ============================================================
set -e

PYTHON_BIN="/mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
export PYTHONNOUSERSITE=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE="outputs/full_run_${TIMESTAMP}"
RAB="../rab_grid_search/data/ringabell_full.txt"
COCO="../prompts/coco/coco_10k.txt"

cd "$(dirname "$0")"
mkdir -p "$BASE"

echo "============================================================"
echo " SHGD Full Experiment Suite"
echo " Output: $BASE"
echo " Start:  $(date)"
echo "============================================================"

# ────────────────────────────────────────────────────────────────
# PHASE 1: ABLATION (7 experiments, GPUs 0-6 parallel)
# ────────────────────────────────────────────────────────────────
echo ""
echo ">>> PHASE 1: Ablation Study (7 experiments, 7 GPUs) <<<"
echo ""

ABL="$BASE/ablation"
mkdir -p "$ABL"

declare -A ABL_CONFIGS
# experiment_name -> "args"
ABL_CONFIGS=(
    # 1. Baseline: no guidance at all (standard SD)
    ["1_baseline"]="--anchor_guidance_scale 0.0 --guide_start_frac 0.0 --guide_end_frac 0.0 --consistency_threshold 0.0"
    # 2. Default: strong guidance in critical window + heal
    ["2_default"]="--config configs/default.yaml"
    # 3. Aggressive: very strong guidance + strong heal
    ["3_aggressive"]="--config configs/aggressive.yaml"
    # 4. Brutal: 미친듯이 강한 guidance + 강한 heal
    ["4_brutal"]="--config configs/brutal.yaml"
    # 5. Wider window: guide beyond critical window [1.0, 0.6]
    ["5_wide_window"]="--anchor_guidance_scale 10.0 --guide_start_frac 1.0 --guide_end_frac 0.6 --heal_strength 0.4"
    # 6. Narrow window: guide only [1.0, 0.9]
    ["6_narrow_window"]="--anchor_guidance_scale 15.0 --guide_start_frac 1.0 --guide_end_frac 0.9 --heal_strength 0.3"
    # 7. Micro heal: per-step correction during guide
    ["7_micro_heal"]="--config configs/micro_heal.yaml"
    # 8. No heal: strong guidance but NO healing (to see heal effect)
    ["8_no_heal"]="--anchor_guidance_scale 15.0 --guide_start_frac 1.0 --guide_end_frac 0.78 --heal_strength 0.0 --consistency_threshold 0.0"
)

# Sort keys for deterministic GPU assignment
SORTED_KEYS=($(echo "${!ABL_CONFIGS[@]}" | tr ' ' '\n' | sort))

PIDS_ABL=()
for i in "${!SORTED_KEYS[@]}"; do
    NAME="${SORTED_KEYS[$i]}"
    ARGS="${ABL_CONFIGS[$NAME]}"
    GPU=$i
    SAVE="$ABL/$NAME"

    # Default config unless overridden in args
    if [[ "$ARGS" != *"--config"* ]]; then
        CFG_ARG="--config configs/default.yaml"
    else
        CFG_ARG=""
    fi

    echo "  [GPU $GPU] $NAME"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN generate.py \
        $CFG_ARG \
        --prompt_file "$RAB" \
        --save_dir "$SAVE" \
        --device cuda:0 \
        --skip_eval \
        $ARGS \
        > "$SAVE.log" 2>&1 &
    PIDS_ABL+=($!)
done

echo ""
echo "  Waiting for ablation (${#PIDS_ABL[@]} jobs)..."
for pid in "${PIDS_ABL[@]}"; do wait $pid; done
echo "  Ablation generation done!"

# ── PRIMARY EVAL: Qwen VLM (8 GPUs parallel) ──
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
echo ""
echo "  [PRIMARY] Running Qwen VLM evaluation on ablation results (8 GPUs parallel)..."
printf "\n%-25s %8s %8s %8s %10s\n" "Experiment" "Safe" "Partial" "Full" "SafeRate"
printf "%-25s %8s %8s %8s %10s\n" "-------------------------" "--------" "--------" "--------" "----------"

VLM_GPU=0
VLM_PIDS=()
VLM_NAMES=()
for NAME in "${SORTED_KEYS[@]}"; do
    SAVE="$ABL/$NAME"
    if [ -d "$SAVE/all" ]; then
        CUDA_VISIBLE_DEVICES=$VLM_GPU PYTHONNOUSERSITE=1 $VLM_PYTHON evaluate.py \
            --image_dir "$SAVE/all" \
            --eval_type qwen \
            --output "$SAVE/eval_qwen.json" > "$SAVE/eval_qwen.log" 2>&1 &
        VLM_PIDS+=($!)
        VLM_NAMES+=("$NAME")
        VLM_GPU=$(( (VLM_GPU + 1) % 8 ))
    fi
done

for i in "${!VLM_PIDS[@]}"; do
    wait "${VLM_PIDS[$i]}"
    NAME="${VLM_NAMES[$i]}"
    SAVE="$ABL/$NAME"
    if [ -f "$SAVE/eval_qwen.json" ]; then
        SAFE=$($PYTHON_BIN -c "import json; d=json.load(open('$SAVE/eval_qwen.json')); s=d.get('qwen',{}).get('summary',{}); print(s.get('safe','?'))" 2>/dev/null || echo "?")
        PARTIAL=$($PYTHON_BIN -c "import json; d=json.load(open('$SAVE/eval_qwen.json')); s=d.get('qwen',{}).get('summary',{}); print(s.get('partial','?'))" 2>/dev/null || echo "?")
        FULL=$($PYTHON_BIN -c "import json; d=json.load(open('$SAVE/eval_qwen.json')); s=d.get('qwen',{}).get('summary',{}); print(s.get('full','?'))" 2>/dev/null || echo "?")
        RATE=$($PYTHON_BIN -c "import json; d=json.load(open('$SAVE/eval_qwen.json')); s=d.get('qwen',{}).get('summary',{}); print(f\"{s.get('safety_rate',0):.2%}\")" 2>/dev/null || echo "?")
        printf "%-25s %8s %8s %8s %10s\n" "$NAME" "$SAFE" "$PARTIAL" "$FULL" "$RATE"
    else
        printf "%-25s %8s\n" "$NAME" "FAILED (check $SAVE/eval_qwen.log)"
    fi
done

# ── SECONDARY EVAL: NudeNet (optional, sequential) ──
echo ""
echo "  [SECONDARY] Running NudeNet evaluation (optional)..."
for NAME in "${SORTED_KEYS[@]}"; do
    SAVE="$ABL/$NAME"
    if [ -d "$SAVE/all" ]; then
        CUDA_VISIBLE_DEVICES=7 $PYTHON_BIN evaluate.py \
            --image_dir "$SAVE/all" \
            --eval_type nudenet \
            --output "$SAVE/eval_nudenet.json" 2>/dev/null || echo "  NudeNet skipped for $NAME"
    fi
done

# ────────────────────────────────────────────────────────────────
# PHASE 2: GRID SEARCH (8 GPUs parallel)
# ────────────────────────────────────────────────────────────────
echo ""
echo ">>> PHASE 2: Grid Search (quick, 8 GPUs) <<<"
echo ""

GRID_DIR="$BASE/grid_search"
$PYTHON_BIN grid_search.py \
    --config configs/default.yaml \
    --prompt_file "$RAB" \
    --output_dir "$GRID_DIR" \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --grid quick

echo "  Grid search done!"

# ────────────────────────────────────────────────────────────────
# PHASE 3: COCO quality eval (best config from grid, 8 GPUs)
# ────────────────────────────────────────────────────────────────
echo ""
echo ">>> PHASE 3: COCO Quality Evaluation (1000 prompts, 8 GPUs) <<<"
echo ""

COCO_DIR="$BASE/coco_default"
COCO_SUBSET="/tmp/coco_1000_${TIMESTAMP}.txt"
head -1000 "$COCO" > "$COCO_SUBSET"

PIDS_COCO=()
for GPU in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN generate.py \
        --config configs/default.yaml \
        --prompt_file "$COCO_SUBSET" \
        --save_dir "$COCO_DIR" \
        --device cuda:0 \
        --gpu_id $GPU \
        --total_gpus 8 \
        --skip_eval \
        > "$COCO_DIR.gpu${GPU}.log" 2>&1 &
    PIDS_COCO+=($!)
done

echo "  Waiting for COCO generation (8 GPUs)..."
for pid in "${PIDS_COCO[@]}"; do wait $pid; done
echo "  COCO generation done!"

# [PRIMARY] Qwen VLM on COCO
echo "  [PRIMARY] Computing Qwen VLM on COCO..."
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $VLM_PYTHON evaluate.py \
    --image_dir "$COCO_DIR/all" \
    --eval_type qwen \
    --output "$COCO_DIR/eval_qwen.json" > "$COCO_DIR/eval_qwen.log" 2>&1 || echo "  (VLM eval failed — check $COCO_DIR/eval_qwen.log)"

# CLIP score
echo "  Computing CLIP Score..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN evaluate.py \
    --image_dir "$COCO_DIR/all" \
    --eval_type clip \
    --prompt_file "$COCO_SUBSET" \
    --output "$COCO_DIR/eval_clip.json" 2>/dev/null || echo "  (CLIP eval skipped)"

# [SECONDARY] NudeNet on COCO (optional)
echo "  [SECONDARY] Computing NudeNet on COCO (optional)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN evaluate.py \
    --image_dir "$COCO_DIR/all" \
    --eval_type nudenet \
    --output "$COCO_DIR/eval_nudenet.json" 2>/dev/null || echo "  (NudeNet eval skipped)"

# ────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " End:    $(date)"
echo " Output: $BASE"
echo "============================================================"
echo ""
echo "Structure:"
echo "  $BASE/"
echo "  ├── ablation/          (8 ablation experiments + NudeNet + VLM eval)"
echo "  ├── grid_search/       (hyperparameter sweep)"
echo "  └── coco_default/      (quality: CLIP + NudeNet + VLM eval)"
echo ""
echo "Evaluation results:"
echo "  - NudeNet: \$EXP/eval_nudenet.json"
echo "  - Qwen VLM: \$EXP/eval_qwen.json"
echo "  - CLIP Score: \$EXP/eval_clip.json"
echo ""
echo "Next steps:"
echo "  1. Check ablation results above (NudeNet + VLM tables)"
echo "  2. Check grid: cat $GRID_DIR/grid_results.json | python -m json.tool | head -50"
