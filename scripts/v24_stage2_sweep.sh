#!/usr/bin/env bash
# =============================================================================
# v24 Stage 2: safety_scale × spatial_threshold sweep
# =============================================================================
# Run AFTER Stage 1 to find optimal ss/st for the best WHERE config.
# Uses Ring-A-Bell (1 sample) for fast iteration.
#
# Usage: bash scripts/v24_stage2_sweep.sh <where_mode> <example_mode> <img_pool> <fusion> <how_mode>
# Example: bash scripts/v24_stage2_sweep.sh noise both cls_multi union anchor_inpaint
# =============================================================================
set -euo pipefail

WHERE=${1:-noise}
EXAMPLE=${2:-both}
POOL=${3:-cls_multi}
FUSION=${4:-union}
HOW=${5:-anchor_inpaint}

REPO="/mnt/home3/yhgil99/unlearning"
P="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
V24="${REPO}/CAS_SpatialCFG/generate_v24.py"
CL="${REPO}/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_full_nudity.pt"
RB="${REPO}/CAS_SpatialCFG/prompts/ringabell.txt"
OUT="${REPO}/CAS_SpatialCFG/outputs/v24_stage2"
LOG="${REPO}/scripts/logs/v24_stage2"

mkdir -p "$LOG" "$OUT"

CLIP_ARG=""
[[ "$EXAMPLE" != "text" ]] && CLIP_ARG="--clip_embeddings ${CL}"

# Sweep grid
SS_VALUES=(0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5)
ST_VALUES=(0.05 0.1 0.15 0.2 0.3)

# Build config list
declare -a CONFIGS=()
for ss in "${SS_VALUES[@]}"; do
  for st in "${ST_VALUES[@]}"; do
    name="${WHERE}_${EXAMPLE}_${HOW}_ss${ss}_st${st}"
    CONFIGS+=("${name}|${ss}|${st}")
  done
done

NUM=${#CONFIGS[@]}
echo "=== v24 Stage 2: ${NUM} configs (${WHERE}/${EXAMPLE}/${POOL}/${FUSION}/${HOW}) ==="
echo "  SS: ${SS_VALUES[*]}"
echo "  ST: ${ST_VALUES[*]}"

# Distribute across 8 GPUs
gpu=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name ss st <<< "$cfg"
  outdir="${OUT}/${name}"

  [ -f "${outdir}/generation_stats.json" ] && { echo "SKIP: $name"; continue; }

  nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} ${P} ${V24} \
    --prompts ${RB} --outdir ${outdir} \
    --where_mode ${WHERE} --example_mode ${EXAMPLE} --img_pool ${POOL} \
    --fusion ${FUSION} --how_mode ${HOW} \
    --safety_scale ${ss} --spatial_threshold ${st} \
    --cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42 \
    ${CLIP_ARG}" > "${LOG}/${name}.log" 2>&1 &

  gpu=$(( (gpu + 1) % 8 ))
done

echo "All ${NUM} configs launched!"
echo ""
echo "Wait ~30 min, then run Qwen eval:"
echo "  bash scripts/v24_stage2_eval.sh"
