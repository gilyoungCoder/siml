#!/bin/bash
# siml-07 GPU 4: FLUX.2-klein SAFREE completion + Qwen eval
set -e
export CUDA_VISIBLE_DEVICES=4
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3"
GEN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_flux2klein_safree.py"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux2klein_experiments/safree"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/flux/siml07_safree_klein"
mkdir -p "$LOG"

echo "=== FLUX.2-klein SAFREE + Eval START $(date) ==="

# --- PART 1: Generate missing SAFREE ---
echo "[GEN] ringabell"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/ringabell" \
  --concept sexual --safree --svf --device cuda:0 > "$LOG/gen_rab.log" 2>&1

echo "[GEN] mja_sexual"
$PY "$GEN" --prompts "$PR/mja_sexual.txt" --outdir "$OUT/mja_sexual" \
  --concept sexual --safree --svf --device cuda:0 > "$LOG/gen_mjas.log" 2>&1

echo "[GEN] mja_violent"
$PY "$GEN" --prompts "$PR/mja_violent.txt" --outdir "$OUT/mja_violent" \
  --concept violence --safree --svf --device cuda:0 > "$LOG/gen_mjav.log" 2>&1

echo "[GEN] mja_disturbing"
$PY "$GEN" --prompts "$PR/mja_disturbing.txt" --outdir "$OUT/mja_disturbing" \
  --concept shocking --safree --svf --device cuda:0 > "$LOG/gen_mjad.log" 2>&1

echo "[GEN] coco_250 (full)"
$PY "$GEN" --prompts "$PR/coco_250.txt" --outdir "$OUT/coco_250" \
  --concept sexual --safree --svf --device cuda:0 > "$LOG/gen_coco.log" 2>&1

echo "[GEN] SAFREE generation DONE $(date)"

# --- PART 2: Qwen eval for SAFREE results ---
cd /mnt/home3/yhgil99/unlearning/vlm
for d in ringabell mma p4dn unlearndiff mja_sexual; do
  echo "[EVAL] SAFREE $d → nudity"
  $VLM_PY "$EVAL" "$OUT/$d" nudity qwen > "$LOG/eval_${d}.log" 2>&1
done

echo "[EVAL] SAFREE mja_violent → violence"
$VLM_PY "$EVAL" "$OUT/mja_violent" violence qwen > "$LOG/eval_mjav.log" 2>&1

echo "[EVAL] SAFREE mja_disturbing → shocking"
$VLM_PY "$EVAL" "$OUT/mja_disturbing" shocking qwen > "$LOG/eval_mjad.log" 2>&1

echo "=== FLUX.2-klein SAFREE ALL DONE $(date) ==="
