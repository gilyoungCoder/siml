#!/bin/bash
# siml-08 GPU 4: FLUX.2-klein 1024x1024 baseline + ours re-generation
set -e
export CUDA_VISIBLE_DEVICES=4
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_flux2klein_v1.py"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
EX="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux2klein_1024"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
LOG="/mnt/home3/yhgil99/unlearning/logs/flux/siml08_klein1024"
mkdir -p "$LOG" "$OUT/baseline" "$OUT/ours"

echo "=== FLUX.2-klein 1024x1024 START $(date) ==="

# --- BASELINE (no safety) ---
for ds in ringabell mma p4dn unlearndiff mja_sexual mja_violent mja_disturbing coco_250; do
  echo "[BL] $ds"
  $PY "$GEN" --prompts "$PR/${ds}.txt" --outdir "$OUT/baseline/$ds" \
    --no_safety --height 1024 --width 1024 --device cuda:0 \
    > "$LOG/bl_${ds}.log" 2>&1
done
echo "[BL] Baseline DONE $(date)"

# --- OURS: key configs ---
# RAB grid
for cas in 0.4 0.6; do
  echo "[OURS] rab_single_ainp_cas${cas}"
  $PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/ours/rab_single_ainp_ss1.0_cas${cas}" \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold $cas \
    --height 1024 --width 1024 --device cuda:0 \
    > "$LOG/ours_rab_ainp_c${cas}.log" 2>&1
done

echo "[OURS] rab_family_ainp_cas0.6"
$PY "$GEN" --prompts "$PR/ringabell.txt" --outdir "$OUT/ours/rab_family_ainp_ss1.0_cas0.6" \
  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 \
  --height 1024 --width 1024 --device cuda:0 \
  > "$LOG/ours_rab_fam.log" 2>&1

# Other datasets (family best config)
for ds in mma p4dn unlearndiff mja_sexual; do
  echo "[OURS] ${ds}_family_ainp"
  $PY "$GEN" --prompts "$PR/${ds}.txt" --outdir "$OUT/ours/${ds}_family_ainp_ss1.0_cas0.6" \
    --family_config "$EX/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 \
    --height 1024 --width 1024 --device cuda:0 \
    > "$LOG/ours_${ds}.log" 2>&1
done

echo "[OURS] mja_violent_family"
$PY "$GEN" --prompts "$PR/mja_violent.txt" --outdir "$OUT/ours/mja_violent_family_ainp_ss1.0_cas0.4" \
  --family_config "$EX/violent/clip_grouped.pt" --family_guidance \
  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 \
  --height 1024 --width 1024 --device cuda:0 \
  > "$LOG/ours_mjav.log" 2>&1

echo "[GEN] Generation DONE $(date)"

# --- QWEN EVAL ---
cd /mnt/home3/yhgil99/unlearning/vlm

echo "[EVAL] Baselines"
for ds in ringabell mma p4dn unlearndiff mja_sexual; do
  $VLM_PY "$EVAL" "$OUT/baseline/$ds" nudity qwen > "$LOG/eval_bl_${ds}.log" 2>&1
done
$VLM_PY "$EVAL" "$OUT/baseline/mja_violent" violence qwen > "$LOG/eval_bl_mjav.log" 2>&1
$VLM_PY "$EVAL" "$OUT/baseline/mja_disturbing" shocking qwen > "$LOG/eval_bl_mjad.log" 2>&1

echo "[EVAL] Ours"
for cfg in rab_single_ainp_ss1.0_cas0.4 rab_single_ainp_ss1.0_cas0.6 rab_family_ainp_ss1.0_cas0.6 \
           mma_family_ainp_ss1.0_cas0.6 p4dn_family_ainp_ss1.0_cas0.6 \
           unlearndiff_family_ainp_ss1.0_cas0.6 mja_sexual_family_ainp_ss1.0_cas0.6; do
  $VLM_PY "$EVAL" "$OUT/ours/$cfg" nudity qwen > "$LOG/eval_ours_${cfg}.log" 2>&1
done
$VLM_PY "$EVAL" "$OUT/ours/mja_violent_family_ainp_ss1.0_cas0.4" violence qwen > "$LOG/eval_ours_mjav.log" 2>&1

echo "=== FLUX.2-klein 1024x1024 ALL DONE $(date) ==="
