#!/bin/bash
# siml-09 GPU 0: FLUX.1-dev ours generation (12 configs sequential)
set -e
export CUDA_VISIBLE_DEVICES=0
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_flux1_v1.py"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
EX="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux1dev_experiments/ours"
LOG="/mnt/home3/yhgil99/unlearning/logs/flux/siml09_flux1dev"
mkdir -p "$LOG"

echo "=== FLUX.1-dev ours generation START $(date) ==="

# 1. RAB grid (6 configs)
echo "[1/12] rab_single_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_single_ainp_ss1.0_cas0.6" \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/rab_single_ainp10c6.log" 2>&1

echo "[2/12] rab_single_ainp_ss1.5_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_single_ainp_ss1.5_cas0.6" \
  --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/rab_single_ainp15c6.log" 2>&1

echo "[3/12] rab_single_hyb_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_single_hyb_ss1.0_cas0.6" \
  --how_mode hybrid --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/rab_single_hyb10c6.log" 2>&1

echo "[4/12] rab_single_ainp_ss1.0_cas0.4"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_single_ainp_ss1.0_cas0.4" \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --device cuda:0 \
  > "$LOG/rab_single_ainp10c4.log" 2>&1

echo "[5/12] rab_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/rab_fam_ainp10c6.log" 2>&1

echo "[6/12] rab_family_ainp_ss1.5_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/rab_family_ainp_ss1.5_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/rab_fam_ainp15c6.log" 2>&1

# 2. Other datasets (family-grouped best config)
echo "[7/12] mma_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/mma.txt" --outdir "$OUT/mma_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/mma_fam.log" 2>&1

echo "[8/12] p4dn_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/p4dn.txt" --outdir "$OUT/p4dn_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/p4dn_fam.log" 2>&1

echo "[9/12] udiff_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/unlearndiff.txt" --outdir "$OUT/udiff_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/udiff_fam.log" 2>&1

echo "[10/12] mja_sexual_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/mja_sexual.txt" --outdir "$OUT/mja_sexual_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/mjas_fam.log" 2>&1

echo "[11/12] mja_violent_family_ainp_ss1.0_cas0.4"
$PY "$GEN" --prompts "$PR/mja_violent.txt" --outdir "$OUT/mja_violent_family_ainp_ss1.0_cas0.4" \
  --family_config "$EX/violent/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --device cuda:0 \
  > "$LOG/mjav_fam.log" 2>&1

echo "[12/12] coco_family_ainp_ss1.0_cas0.6"
$PY "$GEN" --prompts "$PR/coco_250.txt" --outdir "$OUT/coco_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 \
  > "$LOG/coco_fam.log" 2>&1

echo "=== FLUX.1-dev ours ALL DONE $(date) ==="
