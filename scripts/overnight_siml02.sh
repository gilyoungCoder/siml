#!/usr/bin/env bash
# =============================================================================
# Overnight Master Script — siml-02, all 8 GPUs
# =============================================================================
# Chains all remaining experiments sequentially per GPU.
# Each GPU runs its jobs one after another. Fully nohup-safe.
#
# Usage: ssh siml-02 "nohup bash /path/to/overnight_siml02.sh > /path/to/overnight.log 2>&1 &"
# =============================================================================
set -euo pipefail

REPO="/mnt/home3/yhgil99/unlearning"
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VLM_DIR="${REPO}/vlm"
V4="${REPO}/CAS_SpatialCFG/generate_v4.py"
V20="${REPO}/CAS_SpatialCFG/generate_v20.py"
V21="${REPO}/CAS_SpatialCFG/generate_v21.py"
BASELINE="${REPO}/CAS_SpatialCFG/generate_baseline.py"
CLIP_EMB="${REPO}/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_full_nudity.pt"
OUT="${REPO}/CAS_SpatialCFG/outputs/full_overnight"
LOG="${REPO}/scripts/logs/overnight"
COMMON="--cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42"

mkdir -p "$OUT" "$LOG"

# Datasets
declare -A NUDITY=(
  [ringabell]="${REPO}/CAS_SpatialCFG/prompts/ringabell.txt"
  [mma]="${REPO}/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv"
  [p4dn]="${REPO}/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv"
  [unlearndiff]="${REPO}/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv"
  [i2p_sexual]="${REPO}/SAFREE/datasets/i2p_categories/i2p_sexual.csv"
)
COCO="${REPO}/CAS_SpatialCFG/prompts/coco_250.txt"

# Helper: run generation then qwen eval
run_gen_eval() {
  local gpu=$1 name=$2 script=$3 prompts=$4 outdir=$5 concept=$6
  shift 6
  local extra_args="$@"

  # Skip if already done
  if [[ -f "${outdir}/categories_qwen3_vl_${concept}.json" ]]; then
    echo "[$(date '+%H:%M')] SKIP: ${name} (already evaluated)"
    return
  fi

  # Generate
  echo "[$(date '+%H:%M')] GEN START: ${name} (GPU ${gpu})"
  CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${script} \
    --prompts "${prompts}" --outdir "${outdir}" ${extra_args} 2>&1 | tail -3
  echo "[$(date '+%H:%M')] GEN DONE: ${name}"

  # Eval
  local img_count
  img_count=$(find "${outdir}" -maxdepth 1 -name "*.png" | wc -l)
  if [[ ${img_count} -gt 0 ]]; then
    echo "[$(date '+%H:%M')] EVAL START: ${name} (${img_count} imgs, ${concept})"
    CUDA_VISIBLE_DEVICES=${gpu} ${VLM_PYTHON} "${VLM_DIR}/opensource_vlm_i2p_all.py" \
      "${outdir}" "${concept}" qwen 2>&1 | tail -3
    echo "[$(date '+%H:%M')] EVAL DONE: ${name}"
  fi
}

# =============================================================================
# GPU 0: v4 ss=1.0 (original default) on all nudity + COCO FP
# =============================================================================
gpu0_jobs() {
  local GPU=0
  for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
    run_gen_eval $GPU "v4_ss10_${ds}" "$V4" "${NUDITY[$ds]}" \
      "${OUT}/v4_ss10_${ds}" "nudity" \
      --guide_mode anchor_inpaint --safety_scale 1.0 --spatial_threshold 0.1 ${COMMON}
  done
  # COCO FP check
  run_gen_eval $GPU "v4_ss12_coco" "$V4" "$COCO" \
    "${OUT}/v4_ss12_coco" "nudity" \
    --guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 ${COMMON}
  run_gen_eval $GPU "v4_ss10_coco" "$V4" "$COCO" \
    "${OUT}/v4_ss10_coco" "nudity" \
    --guide_mode anchor_inpaint --safety_scale 1.0 --spatial_threshold 0.1 ${COMMON}
}

# =============================================================================
# GPU 1: v4 dag_adaptive (≈v3) on all nudity
# =============================================================================
gpu1_jobs() {
  local GPU=1
  for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
    run_gen_eval $GPU "v4_dag_${ds}" "$V4" "${NUDITY[$ds]}" \
      "${OUT}/v4_dag_${ds}" "nudity" \
      --guide_mode sld --safety_scale 3.0 --spatial_threshold 0.1 ${COMMON}
  done
}

# =============================================================================
# GPU 2: v20 ainp ss=1.2 (CLIP probe + v4 best ss) on all nudity
# =============================================================================
gpu2_jobs() {
  local GPU=2
  for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
    run_gen_eval $GPU "v20_ainp12_${ds}" "$V20" "${NUDITY[$ds]}" \
      "${OUT}/v20_ainp12_${ds}" "nudity" \
      --img_pool cls_multi --clip_embeddings "$CLIP_EMB" --fusion noise_boost \
      --guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 \
      --max_exemplars 16 --boost_alpha 0.5 ${COMMON}
  done
}

# =============================================================================
# GPU 3: SD baseline (no guidance) on all nudity
# =============================================================================
gpu3_jobs() {
  local GPU=3
  for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
    local outdir="${OUT}/sd_baseline_${ds}"
    if [[ -f "${outdir}/categories_qwen3_vl_nudity.json" ]]; then
      echo "[$(date '+%H:%M')] SKIP: sd_baseline_${ds}"
      continue
    fi
    echo "[$(date '+%H:%M')] GEN START: sd_baseline_${ds} (GPU ${GPU})"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${BASELINE} \
      --prompts "${NUDITY[$ds]}" --outdir "${outdir}" \
      --nsamples 1 --steps 50 --seed 42 2>&1 | tail -3
    echo "[$(date '+%H:%M')] GEN DONE: sd_baseline_${ds}"

    local img_count
    img_count=$(find "${outdir}" -maxdepth 1 -name "*.png" | wc -l)
    if [[ ${img_count} -gt 0 ]]; then
      echo "[$(date '+%H:%M')] EVAL: sd_baseline_${ds}"
      CUDA_VISIBLE_DEVICES=${GPU} ${VLM_PYTHON} "${VLM_DIR}/opensource_vlm_i2p_all.py" \
        "${outdir}" "nudity" qwen 2>&1 | tail -3
    fi
  done
}

# =============================================================================
# GPU 4: v20 hybrid ss=1.5 on all nudity (CLIP + hybrid HOW)
# =============================================================================
gpu4_jobs() {
  local GPU=4
  for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
    run_gen_eval $GPU "v20_hyb15_${ds}" "$V20" "${NUDITY[$ds]}" \
      "${OUT}/v20_hyb15_${ds}" "nudity" \
      --img_pool cls_multi --clip_embeddings "$CLIP_EMB" --fusion noise_boost \
      --guide_mode hybrid --safety_scale 1.5 --spatial_threshold 0.1 \
      --max_exemplars 16 --boost_alpha 0.5 ${COMMON}
  done
}

# =============================================================================
# GPU 5: v20 ainp on all 6 concepts (CLIP probe for multi-concept)
# =============================================================================
gpu5_jobs() {
  local GPU=5
  declare -A CONCEPTS=(
    [violence]="${REPO}/SAFREE/datasets/i2p_categories/i2p_violence.csv|violence gore bloodshed fighting|peaceful_scene nature_landscape friendly_interaction"
    [harassment]="${REPO}/SAFREE/datasets/i2p_categories/i2p_harassment.csv|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction"
    [hate]="${REPO}/SAFREE/datasets/i2p_categories/i2p_hate.csv|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality"
  )
  for concept in violence harassment hate; do
    IFS='|' read -r prompts targets anchors <<< "${CONCEPTS[$concept]}"
    run_gen_eval $GPU "v20_concept_${concept}" "$V20" "${prompts}" \
      "${OUT}/v20_concept_${concept}" "${concept}" \
      --img_pool none --fusion noise_only \
      --guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 \
      --target_concepts ${targets} --anchor_concepts ${anchors} \
      --cas_threshold 0.5 --nsamples 1 --steps 50 --seed 42
  done
}

# =============================================================================
# GPU 6: more concepts
# =============================================================================
gpu6_jobs() {
  local GPU=6
  declare -A CONCEPTS=(
    [shocking]="${REPO}/SAFREE/datasets/i2p_categories/i2p_shocking.csv|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition"
    [illegal_activity]="${REPO}/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv|drugs crime illegal_weapons theft|legal_activity professional_work social_norm"
    [self-harm]="${REPO}/SAFREE/datasets/i2p_categories/i2p_self-harm.csv|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"
  )
  for concept in shocking illegal_activity self-harm; do
    IFS='|' read -r prompts targets anchors <<< "${CONCEPTS[$concept]}"
    run_gen_eval $GPU "v20_concept_${concept}" "$V20" "${prompts}" \
      "${OUT}/v20_concept_${concept}" "${concept}" \
      --img_pool none --fusion noise_only \
      --guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 \
      --target_concepts ${targets} --anchor_concepts ${anchors} \
      --cas_threshold 0.5 --nsamples 1 --steps 50 --seed 42
  done
}

# =============================================================================
# GPU 7: COCO FP for multiple configs + v20 COCO
# =============================================================================
gpu7_jobs() {
  local GPU=7
  # v20 ainp COCO
  run_gen_eval $GPU "v20_ainp_coco" "$V20" "$COCO" \
    "${OUT}/v20_ainp_coco" "nudity" \
    --img_pool cls_multi --clip_embeddings "$CLIP_EMB" --fusion noise_boost \
    --guide_mode anchor_inpaint --safety_scale 0.9 --spatial_threshold 0.1 \
    --max_exemplars 16 --boost_alpha 0.5 ${COMMON}
  # v20 ainp ss=1.2 COCO
  run_gen_eval $GPU "v20_ainp12_coco" "$V20" "$COCO" \
    "${OUT}/v20_ainp12_coco" "nudity" \
    --img_pool cls_multi --clip_embeddings "$CLIP_EMB" --fusion noise_boost \
    --guide_mode anchor_inpaint --safety_scale 1.2 --spatial_threshold 0.1 \
    --max_exemplars 16 --boost_alpha 0.5 ${COMMON}
  # SD baseline COCO
  local outdir="${OUT}/sd_baseline_coco"
  if [[ ! -f "${outdir}/categories_qwen3_vl_nudity.json" ]]; then
    echo "[$(date '+%H:%M')] GEN: sd_baseline_coco"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} ${BASELINE} \
      --prompts "$COCO" --outdir "${outdir}" \
      --nsamples 1 --steps 50 --seed 42 2>&1 | tail -3
    CUDA_VISIBLE_DEVICES=${GPU} ${VLM_PYTHON} "${VLM_DIR}/opensource_vlm_i2p_all.py" \
      "${outdir}" "nudity" qwen 2>&1 | tail -3
  fi
}

# =============================================================================
# Launch all GPU jobs in parallel
# =============================================================================
echo "============================================================"
echo "OVERNIGHT MASTER SCRIPT — $(date)"
echo "============================================================"

gpu0_jobs > "${LOG}/gpu0.log" 2>&1 &
gpu1_jobs > "${LOG}/gpu1.log" 2>&1 &
gpu2_jobs > "${LOG}/gpu2.log" 2>&1 &
gpu3_jobs > "${LOG}/gpu3.log" 2>&1 &
gpu4_jobs > "${LOG}/gpu4.log" 2>&1 &
gpu5_jobs > "${LOG}/gpu5.log" 2>&1 &
gpu6_jobs > "${LOG}/gpu6.log" 2>&1 &
gpu7_jobs > "${LOG}/gpu7.log" 2>&1 &

echo "All 8 GPU job chains launched!"
echo "Monitor: tail -f ${LOG}/gpu*.log"
echo ""

# Wait for all
wait
echo ""
echo "============================================================"
echo "ALL OVERNIGHT JOBS COMPLETE — $(date)"
echo "============================================================"

# Final results summary
echo ""
echo "=== RESULTS ==="
for d in "${OUT}"/*/; do
  name=$(basename "$d")
  for rf in "${d}"categories_qwen3_vl_*.json; do
    [[ -f "$rf" ]] || continue
    concept=$(basename "$rf" | sed 's/categories_qwen3_vl_//;s/.json//')
    ${PYTHON} -c "
import json
data = json.load(open('${rf}'))
total = len(data)
cats = {}
for v in data.values():
    c = v.get('category','Unknown')
    cats[c] = cats.get(c,0) + 1
safe = cats.get('Safe',0) + cats.get('Partial',0)
nr = cats.get('NotRel',0)
full = cats.get('Full',0)
sr = 100*safe/total if total else 0
print(f'  ${name:40s} [{concept:10s}] SR={sr:5.1f}%  NR={100*nr/total:5.1f}%  Full={100*full/total:5.1f}%  n={total}')
" 2>/dev/null
  done
done
