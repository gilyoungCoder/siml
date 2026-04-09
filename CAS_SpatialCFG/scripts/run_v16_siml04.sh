#!/bin/bash
# ============================================================================
# v16 Grid Search + Evaluation (Contrastive Image Direction)
# WHEN: CAS (tau=0.6) | WHERE: contrastive direction probe | HOW: dag_adaptive/hybrid
# probe_source: text / image / both
# Step 0: prepare_contrastive_direction.py (1회)
# ============================================================================
set -uo pipefail

GPU_IDS=(1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../scripts/lib/repo_env.sh"
cd "$SCRIPT_DIR"

PROMPTS="prompts/ringabell.txt"
CONCEPT_DIR="exemplars/sd14/concept_directions.pt"
CONTRASTIVE_EMBED="exemplars/sd14/contrastive_embeddings.pt"
OUTPUT_BASE="outputs/v16"
LOG_DIR="logs/v16"
mkdir -p "$LOG_DIR"

PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
PYTHON_VLM="${UNLEARNING_VLM_PYTHON}"

CAS_THRESHOLD=0.6
NUM_SAMPLES=4
STEPS=50
CFG_SCALE=7.5
SEED=42

# ===== STEP 0: PREP =====
if [[ ! -f "$CONTRASTIVE_EMBED" ]]; then
  echo "=== Step 0: Preparing contrastive direction ==="
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} $PYTHON prepare_contrastive_direction.py \
    --exemplar_dir exemplars/sd14/exemplar_images \
    --output "$CONTRASTIVE_EMBED" \
    > "$LOG_DIR/prep.log" 2>&1
  echo "  Contrastive embeddings saved to $CONTRASTIVE_EMBED"
else
  echo "=== Step 0: SKIP (contrastive embeddings exist) ==="
fi

# ===== GRID =====
GUIDE_MODES=(dag_adaptive hybrid)
SAFETY_SCALES=(1.0 2.0 3.0 5.0)
SPATIAL_THRESHOLDS=(0.2 0.3 0.4)
PROBE_SOURCES=(text image both)
CONTRASTIVE_MODES=(cls patch mixed)
WHERE_MODES=(probe_only fused)

# contrastive_mode는 image/both에서만 의미 → text는 cls 1개만
declare -a CONFIGS=()
for gm in "${GUIDE_MODES[@]}"; do
  for ss in "${SAFETY_SCALES[@]}"; do
    for st in "${SPATIAL_THRESHOLDS[@]}"; do
      for ps in "${PROBE_SOURCES[@]}"; do
        for wm in "${WHERE_MODES[@]}"; do
          if [[ "$ps" == "text" ]]; then
            tag="${ps}_${gm}_ss${ss}_st${st}_${wm}"
            CONFIGS+=("${tag}|${gm}|${ss}|${st}|${ps}|cls|${wm}")
          else
            for cm in "${CONTRASTIVE_MODES[@]}"; do
              tag="${ps}_${cm}_${gm}_ss${ss}_st${st}_${wm}"
              CONFIGS+=("${tag}|${gm}|${ss}|${st}|${ps}|${cm}|${wm}")
            done
          fi
        done
      done
    done
  done
done

TOTAL=${#CONFIGS[@]}
echo "========================================="
echo " v16 Grid Search: $TOTAL configs"
echo " GPUs: ${GPU_IDS[*]}"
echo "========================================="

run_one() {
  local gpu_id=$1
  local config_str=$2
  IFS='|' read -r tag gm ss st ps cm wm <<< "$config_str"
  local out_dir="${OUTPUT_BASE}/ringabell_${tag}"

  if [[ -f "$out_dir/stats.json" ]]; then
    echo "  [GPU $gpu_id] SKIP: $tag"
    return 0
  fi

  echo "  [GPU $gpu_id] $tag"

  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON generate_v16.py \
    --prompts "$PROMPTS" \
    --outdir "$out_dir" \
    --guide_mode "$gm" \
    --safety_scale "$ss" \
    --spatial_threshold "$st" \
    --cas_threshold "$CAS_THRESHOLD" \
    --probe_source "$ps" \
    --contrastive_mode "$cm" \
    --where_mode "$wm" \
    --contrastive_embeddings "$CONTRASTIVE_EMBED" \
    --concept_dir_path "$CONCEPT_DIR" \
    --nsamples "$NUM_SAMPLES" \
    --steps "$STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --seed "$SEED" \
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
echo " v16 Grid Search COMPLETE: $TOTAL configs"
echo "========================================="

$PYTHON ../scripts/aggregate_nudity_results.py "${OUTPUT_BASE}"/ringabell_* \
  > "$LOG_DIR/aggregate.log" 2>&1 || true
