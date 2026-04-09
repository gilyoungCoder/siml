#!/bin/bash
# ============================================================================
# v15 Grid Search + Evaluation (CLIP Patch Token Probe)
# WHEN: CAS (tau=0.6) | WHERE: CLIP patch token crossattn probe | HOW: dag_adaptive/hybrid
# probe_source: text / image / both
# Step 0: prepare_clip_patch_tokens.py (1회)
# ============================================================================
set -uo pipefail

GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../scripts/lib/repo_env.sh"
cd "$SCRIPT_DIR"

PROMPTS="prompts/ringabell.txt"
CONCEPT_DIR="exemplars/sd14/concept_directions.pt"
PATCH_TOKENS="exemplars/sd14/clip_patch_tokens.pt"
OUTPUT_BASE="outputs/v15"
LOG_DIR="logs/v15"
mkdir -p "$LOG_DIR"

PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
PYTHON_VLM="${UNLEARNING_VLM_PYTHON}"

CAS_THRESHOLD=0.6
NUM_SAMPLES=4
STEPS=50
CFG_SCALE=7.5
SEED=42

# ===== STEP 0: PREP (1회만) =====
if [[ ! -f "$PATCH_TOKENS" ]]; then
  echo "=== Step 0: Preparing CLIP patch tokens ==="
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} $PYTHON prepare_clip_patch_tokens.py \
    --exemplar_dir exemplars/sd14/exemplar_images \
    --output "$PATCH_TOKENS" \
    --num_patches 16 \
    > "$LOG_DIR/prep.log" 2>&1
  echo "  Patch tokens saved to $PATCH_TOKENS"
else
  echo "=== Step 0: SKIP (patch tokens exist) ==="
fi

# ===== GRID =====
GUIDE_MODES=(dag_adaptive hybrid)
SAFETY_SCALES=(1.0 2.0 3.0 5.0)
SPATIAL_THRESHOLDS=(0.2 0.3 0.4)
PROBE_SOURCES=(text image both)
NUM_PATCHES_LIST=(16)  # 기본 16, 추가 sweep은 별도

declare -a CONFIGS=()
for gm in "${GUIDE_MODES[@]}"; do
  for ss in "${SAFETY_SCALES[@]}"; do
    for st in "${SPATIAL_THRESHOLDS[@]}"; do
      for ps in "${PROBE_SOURCES[@]}"; do
        for np in "${NUM_PATCHES_LIST[@]}"; do
          tag="${ps}_${gm}_ss${ss}_st${st}_np${np}"
          CONFIGS+=("${tag}|${gm}|${ss}|${st}|${ps}|${np}")
        done
      done
    done
  done
done

TOTAL=${#CONFIGS[@]}
echo "========================================="
echo " v15 Grid Search: $TOTAL configs"
echo " GPUs: ${GPU_IDS[*]}"
echo "========================================="

run_one() {
  local gpu_id=$1
  local config_str=$2
  IFS='|' read -r tag gm ss st ps np <<< "$config_str"
  local out_dir="${OUTPUT_BASE}/ringabell_${tag}"

  if [[ -f "$out_dir/stats.json" ]]; then
    echo "  [GPU $gpu_id] SKIP: $tag"
    return 0
  fi

  echo "  [GPU $gpu_id] $tag"

  local gen_args=(
    --prompts "$PROMPTS"
    --outdir "$out_dir"
    --guide_mode "$gm"
    --safety_scale "$ss"
    --spatial_threshold "$st"
    --cas_threshold "$CAS_THRESHOLD"
    --probe_source "$ps"
    --num_patches "$np"
    --concept_dir_path "$CONCEPT_DIR"
    --patch_embeddings "$PATCH_TOKENS"
    --nsamples "$NUM_SAMPLES"
    --steps "$STEPS"
    --cfg_scale "$CFG_SCALE"
    --seed "$SEED"
  )

  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON generate_v15.py "${gen_args[@]}" \
    > "$LOG_DIR/${tag}_gen.log" 2>&1

  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON ../vlm/eval_nudenet.py "$out_dir" --threshold 0.8 \
    > "$LOG_DIR/${tag}_nn.log" 2>&1

  local nn_result=$(grep -oP 'Unsafe rate: \K[0-9.]+' "$out_dir/results_nudenet_08.txt" 2>/dev/null || echo "N/A")

  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_VLM ../vlm/opensource_vlm_nudity.py "$out_dir" qwen \
    > "$LOG_DIR/${tag}_qwen.log" 2>&1

  echo "  [GPU $gpu_id] DONE: $tag -> NN=${nn_result}%"
}

running=0
config_idx=0
for config_str in "${CONFIGS[@]}"; do
  gpu_idx=$((config_idx % NUM_GPUS))
  gpu_id=${GPU_IDS[$gpu_idx]}
  run_one "$gpu_id" "$config_str" &
  running=$((running + 1))
  config_idx=$((config_idx + 1))
  if (( running >= NUM_GPUS )); then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

echo ""
echo "========================================="
echo " v15 Grid Search COMPLETE: $TOTAL configs"
echo "========================================="

$PYTHON ../scripts/aggregate_nudity_results.py "${OUTPUT_BASE}"/ringabell_* \
  > "$LOG_DIR/aggregate.log" 2>&1 || true
