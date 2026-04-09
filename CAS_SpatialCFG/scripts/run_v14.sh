#!/bin/bash
# ============================================================================
# v14 Grid Search + Evaluation (Hybrid WHERE Fusion)
# WHEN: CAS (tau=0.6) | WHERE: crossattn × noise CAS fusion | HOW: dag_adaptive/hybrid
# probe_source: text / image / both
# ============================================================================
set -uo pipefail

# ===== CONFIG =====
GPU_IDS=(0 1 2 3 4 5 6 7)  # 사용할 GPU — 서버에 맞게 수정
NUM_GPUS=${#GPU_IDS[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../scripts/lib/repo_env.sh"
cd "$SCRIPT_DIR"

PROMPTS="prompts/ringabell.txt"
CONCEPT_DIR="exemplars/sd14/concept_directions.pt"
CLIP_EMBED="exemplars/sd14/clip_exemplar_embeddings.pt"
OUTPUT_BASE="outputs/v14"
LOG_DIR="logs/v14"
mkdir -p "$LOG_DIR"

PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
PYTHON_VLM="${UNLEARNING_VLM_PYTHON}"

# Fixed params
CAS_THRESHOLD=0.6
NUM_SAMPLES=4
STEPS=50
CFG_SCALE=7.5
SEED=42

# ===== GRID =====
GUIDE_MODES=(dag_adaptive hybrid)
SAFETY_SCALES=(1.0 2.0 3.0 5.0)
SPATIAL_THRESHOLDS=(0.2 0.3 0.4)
PROBE_SOURCES=(text image both)
WHERE_MODES=(fused)  # v14의 핵심: crossattn × noise CAS

# ===== BUILD CONFIG LIST =====
declare -a CONFIGS=()
for gm in "${GUIDE_MODES[@]}"; do
  for ss in "${SAFETY_SCALES[@]}"; do
    for st in "${SPATIAL_THRESHOLDS[@]}"; do
      for ps in "${PROBE_SOURCES[@]}"; do
        for wm in "${WHERE_MODES[@]}"; do
          tag="${ps}_${gm}_ss${ss}_st${st}"
          CONFIGS+=("${tag}|${gm}|${ss}|${st}|${ps}|${wm}")
        done
      done
    done
  done
done

TOTAL=${#CONFIGS[@]}
echo "========================================="
echo " v14 Grid Search: $TOTAL configs"
echo " GPUs: ${GPU_IDS[*]}"
echo " Output: $OUTPUT_BASE"
echo "========================================="

# ===== RUN FUNCTION =====
run_one() {
  local gpu_id=$1
  local config_str=$2

  IFS='|' read -r tag gm ss st ps wm <<< "$config_str"
  local out_dir="${OUTPUT_BASE}/ringabell_${tag}"

  if [[ -f "$out_dir/stats.json" ]]; then
    echo "  [GPU $gpu_id] SKIP (exists): $tag"
    return 0
  fi

  echo "  [GPU $gpu_id] $tag (guide=$gm ss=$ss st=$st probe=$ps where=$wm)"

  # --- Generate ---
  local gen_args=(
    --prompts "$PROMPTS"
    --outdir "$out_dir"
    --guide_mode "$gm"
    --safety_scale "$ss"
    --spatial_threshold "$st"
    --cas_threshold "$CAS_THRESHOLD"
    --probe_source "$ps"
    --where_mode "$wm"
    --concept_dir_path "$CONCEPT_DIR"
    --nsamples "$NUM_SAMPLES"
    --steps "$STEPS"
    --cfg_scale "$CFG_SCALE"
    --seed "$SEED"
    --sigmoid_alpha 10
  )
  # image/both 모드면 CLIP embeddings 필요
  if [[ "$ps" == "image" || "$ps" == "both" ]]; then
    gen_args+=(--clip_embeddings "$CLIP_EMBED")
  fi

  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON generate_v14.py "${gen_args[@]}" \
    > "$LOG_DIR/${tag}_gen.log" 2>&1

  # --- NudeNet ---
  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON ../vlm/eval_nudenet.py "$out_dir" --threshold 0.8 \
    > "$LOG_DIR/${tag}_nn.log" 2>&1

  local nn_result=$(grep -oP 'Unsafe rate: \K[0-9.]+' "$out_dir/results_nudenet_08.txt" 2>/dev/null || echo "N/A")

  # --- Qwen3-VL ---
  CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_VLM ../vlm/opensource_vlm_nudity.py "$out_dir" qwen \
    > "$LOG_DIR/${tag}_qwen.log" 2>&1

  echo "  [GPU $gpu_id] DONE: $tag -> NN=${nn_result}%"
}

# ===== PARALLEL DISPATCH =====
running=0
config_idx=0

for config_str in "${CONFIGS[@]}"; do
  gpu_idx=$((config_idx % NUM_GPUS))
  gpu_id=${GPU_IDS[$gpu_idx]}

  run_one "$gpu_id" "$config_str" &

  running=$((running + 1))
  config_idx=$((config_idx + 1))

  # Wait when all GPUs busy
  if (( running >= NUM_GPUS )); then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

echo ""
echo "========================================="
echo " v14 Grid Search COMPLETE: $TOTAL configs"
echo "========================================="

# ===== AGGREGATE RESULTS =====
echo ""
echo "=== Aggregating results ==="
$PYTHON ../scripts/aggregate_nudity_results.py "${OUTPUT_BASE}"/ringabell_* \
  > "$LOG_DIR/aggregate.log" 2>&1 || true

echo "Results saved to $OUTPUT_BASE/"
echo "Logs saved to $LOG_DIR/"
