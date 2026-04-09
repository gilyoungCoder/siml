#!/bin/bash
# ============================================================================
# SIML-04: 남은 실험 전체 처리
# GPU 1-7 사용 (GPU 0은 다른 사용자)
# 1) v14/v15/v17/v18 기존 output에 Qwen eval 추가
# 2) v16 prep + grid search + eval (처음부터)
# 3) v19 (SIML-05에서 미완이면)
# ============================================================================
set -uo pipefail

GPU_IDS=(1 2 3 4 5 6 7)  # GPU 0 제외!
NUM_GPUS=${#GPU_IDS[@]}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../scripts/lib/repo_env.sh"
cd "$SCRIPT_DIR"

PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
PYTHON_VLM="${UNLEARNING_VLM_PYTHON}"
LOG_DIR="logs/remaining"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " SIML-04 Remaining Jobs"
echo " GPUs: ${GPU_IDS[*]} (GPU 0 excluded)"
echo " Started: $(date)"
echo "============================================"

# ============================================================
# PART 1: Qwen eval on all existing v14/v15/v17/v18 outputs
# ============================================================
echo ""
echo ">>> PART 1: Qwen eval on existing outputs"

# Collect all dirs that have stats.json but no qwen results
declare -a QWEN_DIRS=()
for ver in v14 v15 v17 v18; do
  for d in outputs/$ver/ringabell_*/; do
    if [[ -f "$d/stats.json" ]] && ! unlearning_find_qwen_result_txt "$d" >/dev/null 2>&1; then
      QWEN_DIRS+=("$d")
    fi
  done
done

TOTAL_QWEN=${#QWEN_DIRS[@]}
echo "  $TOTAL_QWEN directories need Qwen eval"

qwen_idx=0
running=0
for d in "${QWEN_DIRS[@]}"; do
  gpu_idx=$((qwen_idx % NUM_GPUS))
  gpu_id=${GPU_IDS[$gpu_idx]}
  tag=$(basename "$d")

  (
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_VLM ../vlm/opensource_vlm_nudity.py "$d" qwen \
      > "$LOG_DIR/${tag}_qwen.log" 2>&1
    echo "  [GPU $gpu_id] QWEN DONE: $tag"
  ) &

  running=$((running + 1))
  qwen_idx=$((qwen_idx + 1))

  if (( running >= NUM_GPUS )); then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "  Qwen eval complete: $TOTAL_QWEN dirs processed"

# ============================================================
# PART 2: v16 (contrastive direction) — prep + grid search
# ============================================================
echo ""
echo ">>> PART 2: v16 Grid Search + Eval"

# Step 0: Prep contrastive embeddings
CONTRASTIVE_EMBED="exemplars/sd14/contrastive_embeddings.pt"
if [[ ! -f "$CONTRASTIVE_EMBED" ]]; then
  echo "  Preparing contrastive embeddings..."
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} $PYTHON prepare_contrastive_direction.py \
    --target_dir exemplars/sd14/exemplar_images \
    --anchor_dir exemplars/sd14/exemplar_images \
    --output "$CONTRASTIVE_EMBED" \
    > "$LOG_DIR/v16_prep.log" 2>&1
  echo "  Saved to $CONTRASTIVE_EMBED"
else
  echo "  SKIP: contrastive embeddings already exist"
fi

# Run v16 grid search with fixed GPU list
export GPU_IDS_OVERRIDE="1 2 3 4 5 6 7"
# Inline v16 grid search since run_v16.sh uses GPU_IDS=(0..7)
bash scripts/run_v16_siml04.sh
echo "  v16 complete"

# ============================================================
# PART 3: v19 if not done
# ============================================================
V19_DONE=$(ls outputs/v19/*/stats.json 2>/dev/null | wc -l)
if (( V19_DONE == 0 )); then
  echo ""
  echo ">>> PART 3: v19 Grid Search + Eval"
  bash scripts/run_v19_siml04.sh
  echo "  v19 complete"
else
  echo ""
  echo ">>> PART 3: v19 SKIP ($V19_DONE configs already done)"
fi

echo ""
echo "============================================"
echo " SIML-04 ALL COMPLETE"
echo " Finished: $(date)"
echo "============================================"
