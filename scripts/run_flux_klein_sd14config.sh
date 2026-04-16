#!/bin/bash
# FLUX.2-klein 512 with SD1.4 best family config: both ss1.2 at0.1 cas0.6
# All 5 nudity datasets sequential
set -e
export CUDA_VISIBLE_DEVICES=0
PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
GEN="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_flux2klein_v1.py"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"
PR="/mnt/home3/yhgil99/unlearning/SafeGen/prompts"
EX="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/flux2klein_sd14cfg"
LOG="/mnt/home3/yhgil99/unlearning/logs/flux_klein_sd14cfg"
mkdir -p "$OUT" "$LOG"

echo "=== FLUX.2-klein 512 with SD1.4 best config START $(date) ==="

# Config: both + ss1.2 + at0.1 (cas_threshold 0.6 default, family_guidance)
for ds in ringabell mma p4dn unlearndiff mja_sexual; do
  outdir="$OUT/${ds}_both_fam_ss1.2_at0.1_cas0.6"
  [ -f "$outdir/generation_stats.json" ] && { echo "[SKIP] $ds"; continue; }

  echo "[GEN] $ds"
  # Note: generate_flux2klein_v1.py may not support attn_threshold, check
  $PY "$GEN" --prompts "$PR/${ds}.txt" --outdir "$outdir" \
    --how_mode anchor_inpaint --safety_scale 1.2 --cas_threshold 0.6 \
    --family_config "$EX" --family_guidance \
    --device cuda:0 \
    > "$LOG/${ds}.log" 2>&1

  echo "[EVAL] $ds"
  cd /mnt/home3/yhgil99/unlearning/vlm
  $VLM_PY "$EVAL" "$outdir" nudity qwen > "$LOG/${ds}_eval.log" 2>&1
  grep "SR " "$outdir/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "eval pending"
done

echo "=== FLUX.2-klein sd14-config ALL DONE $(date) ==="
