#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 2026-04-09: siml-02 전체 GPU 활용 — 평가 + 추가 실험
# GPU 4 사용중 (illegal_activity generation), 나머지 0,1,2,3,5,6,7 사용
# =============================================================================

VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
CLIP=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
ON=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_overnight_final
MEGA=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega
REPO=/mnt/home3/yhgil99/unlearning

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  if [ -f "${dir}/categories_qwen3_vl_${concept}.json" ]; then
    echo "[SKIP] Already evaluated: $(basename $dir) / $concept"
    return
  fi
  local nimgs=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $nimgs -lt 10 ]; then
    echo "[SKIP] Too few images ($nimgs): $(basename $dir)"
    return
  fi
  echo "[$(date +%H:%M)] GPU $gpu: Evaluating $(basename $dir) / $concept ($nimgs imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -3
}

# =============================================================================
# GPU 0: multi_nude_harassment_ss1.0 — nudity + harassment eval (2472 imgs)
# =============================================================================
(
echo "=== GPU 0: multi_nude_harassment_ss1.0 ==="
eval_qwen 0 "$ON/multi_nude_harassment_ss1.0" nudity
eval_qwen 0 "$ON/multi_nude_harassment_ss1.0" harassment
echo "GPU0 DONE — $(date)"
) &

# =============================================================================
# GPU 1: multi_nude_harassment_ss1.2 — nudity + harassment eval (2472 imgs)
# =============================================================================
(
echo "=== GPU 1: multi_nude_harassment_ss1.2 ==="
eval_qwen 1 "$ON/multi_nude_harassment_ss1.2" nudity
eval_qwen 1 "$ON/multi_nude_harassment_ss1.2" harassment
echo "GPU1 DONE — $(date)"
) &

# =============================================================================
# GPU 2: multi_nude_shocking_ss1.0 — nudity + shocking eval (2568 imgs)
# =============================================================================
(
echo "=== GPU 2: multi_nude_shocking_ss1.0 ==="
eval_qwen 2 "$ON/multi_nude_shocking_ss1.0" nudity
eval_qwen 2 "$ON/multi_nude_shocking_ss1.0" shocking
echo "GPU2 DONE — $(date)"
) &

# =============================================================================
# GPU 3: multi_nude_shocking_ss1.2 — nudity + shocking eval (2568 imgs)
# =============================================================================
(
echo "=== GPU 3: multi_nude_shocking_ss1.2 ==="
eval_qwen 3 "$ON/multi_nude_shocking_ss1.2" nudity
eval_qwen 3 "$ON/multi_nude_shocking_ss1.2" shocking
echo "GPU3 DONE — $(date)"
) &

# =============================================================================
# GPU 5: multi_nude_hate — nudity + hate eval (693×2 imgs)
# =============================================================================
(
echo "=== GPU 5: multi_nude_hate ==="
eval_qwen 5 "$ON/multi_nude_hate_ss1.0" nudity
eval_qwen 5 "$ON/multi_nude_hate_ss1.0" hate
eval_qwen 5 "$ON/multi_nude_hate_ss1.2" nudity
eval_qwen 5 "$ON/multi_nude_hate_ss1.2" hate
echo "GPU5 DONE — $(date)"
) &

# =============================================================================
# GPU 6: v27 COCO FP (generate hyb + eval) + warhol/vg style evals
# =============================================================================
(
echo "=== GPU 6: COCO FP + style evals ==="
# Generate COCO with hyb best config (if not exists)
COCO_OUT=$MEGA/nude_hyb_coco
if [ ! -f "${COCO_OUT}/generation_stats.json" ]; then
  echo "[$(date +%H:%M)] GPU 6: Generating COCO hyb..."
  CUDA_VISIBLE_DEVICES=6 $P $V27 \
    --prompts "$REPO/CAS_SpatialCFG/prompts/coco_250.txt" \
    --outdir "$COCO_OUT" \
    --nsamples 1 --steps 50 --seed 42 \
    --cas_threshold 0.6 \
    --probe_mode both --clip_embeddings $CLIP \
    --attn_threshold 0.1 --img_attn_threshold 0.4 \
    --how_mode hybrid --target_scale 15 --anchor_scale 15 \
    2>&1 | tail -3
fi
eval_qwen 6 "$COCO_OUT" nudity

# Evaluate warhol/vg style experiments
for d in $ON/multi_warhol_nude_*; do
  [ -d "$d" ] && eval_qwen 6 "$d" nudity
done
for d in $ON/multi_vg_nude_*; do
  [ -d "$d" ] && eval_qwen 6 "$d" nudity
done
echo "GPU6 DONE — $(date)"
) &

# =============================================================================
# GPU 7: Remaining multi-concept evals + NudeNet batch
# =============================================================================
(
echo "=== GPU 7: Remaining evals ==="
# Evaluate any remaining overnight_final dirs
for d in $ON/nude_*; do
  [ -d "$d" ] && eval_qwen 7 "$d" nudity
done
for d in $ON/c2_violence_*; do
  [ -d "$d" ] && eval_qwen 7 "$d" violence
done
# v27_final multi-concept dirs that might be missing evals
for d in $MEGA/nude_hyb_*; do
  [ -d "$d" ] && eval_qwen 7 "$d" nudity
done
echo "GPU7 DONE — $(date)"
) &

echo "=============================================="
echo "  ALL EVAL LAUNCHED — siml-02 — $(date)"
echo "  GPU 0: multi_nude_harassment_ss1.0 (2472)"
echo "  GPU 1: multi_nude_harassment_ss1.2 (2472)"
echo "  GPU 2: multi_nude_shocking_ss1.0 (2568)"
echo "  GPU 3: multi_nude_shocking_ss1.2 (2568)"
echo "  GPU 5: multi_nude_hate (693×2)"
echo "  GPU 6: COCO hyb FP + style evals"
echo "  GPU 7: Remaining evals + NudeNet"
echo "=============================================="

wait
echo "ALL EVAL DONE — $(date)"
