#!/bin/bash
# siml-03 GPU 0,1: Re-evaluate ALL nudity datasets with OLD (liberal) Qwen prompt
# Uses v1_backup eval script
set -e
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v1_backup.py"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"
LOG="/mnt/home3/yhgil99/unlearning/logs/old_eval_siml03"
mkdir -p "$LOG"

cd /mnt/home3/yhgil99/unlearning/vlm

echo "=== OLD EVAL (liberal Qwen) START $(date) ==="

# Output files will be named results_qwen3_vl_nudity_old.txt to avoid overwriting

# --- GPU 0: Baseline + SAFREE ---
(
export CUDA_VISIBLE_DEVICES=0

# Baseline
for ds in rab mma p4dn unlearndiff mja_sexual mja_violent mja_disturbing; do
  d="$BASE/baselines_v2/$ds"
  [ -d "$d" ] && {
    echo "[GPU0] baseline $ds"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/bl_${ds}.log" 2>&1
    # Rename output to _old to preserve
    for f in "$d"/results_qwen3_vl_nudity.txt "$d"/categories_qwen3_vl_nudity.json; do
      [ -f "$f" ] && cp "$f" "${f%.txt}_old.txt" 2>/dev/null; cp "$f" "${f%.json}_old.json" 2>/dev/null
    done
  }
done

# SAFREE
for ds in rab mma p4dn unlearndiff mja_sexual; do
  d="$BASE/safree_reproduction/$ds/all"
  [ ! -d "$d" ] && d="$BASE/safree_reproduction/$ds"
  [ -d "$d" ] && {
    echo "[GPU0] safree $ds"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/safree_${ds}.log" 2>&1
  }
done

echo "[GPU0] DONE $(date)"
) &
PID0=$!

# --- GPU 1: Ours best configs ---
(
export CUDA_VISIBLE_DEVICES=1

# v2 ours best per dataset
declare -A OURS_DIRS=(
  ["rab_best"]="$BASE/v2_experiments/sexual/rab_text_anchor_inpaint_single_cas0.4_ss1.2"
  ["rab_fam"]="$BASE/v2_experiments/sexual/rab_text_anchor_inpaint_fam_cas0.4_ss1.2"
  ["rab_both"]="$BASE/v2_experiments/sexual/rab_both_anchor_inpaint_single_cas0.4_ss1.2"
  ["mma_best"]="$BASE/v2_experiments/sexual/mma_both_anchor_inpaint_fam_cas0.6_ss1.2"
  ["p4dn_best"]="$BASE/v2_experiments/sexual/p4dn_both_anchor_inpaint_single_cas0.6_ss1.2"
  ["udiff_best"]="$BASE/v2_experiments/sexual/udiff_both_anchor_inpaint_single_cas0.6_ss1.2"
  ["mja_sex_best"]="$BASE/v2_experiments/sexual/mja_both_anchor_inpaint_fam_cas0.6_ss1.2"
)

for key in rab_best rab_fam rab_both mma_best p4dn_best udiff_best mja_sex_best; do
  d="${OURS_DIRS[$key]}"
  [ -d "$d" ] && {
    echo "[GPU1] ours $key"
    $VLM_PY "$EVAL" "$d" nudity qwen > "$LOG/ours_${key}.log" 2>&1
  }
done

echo "[GPU1] DONE $(date)"
) &
PID1=$!

wait $PID0 $PID1
echo "=== OLD EVAL ALL DONE $(date) ==="

# --- Collect results ---
echo ""
echo "========== RESULTS (OLD liberal Qwen) =========="
echo ""
echo "--- Baseline ---"
for ds in rab mma p4dn unlearndiff mja_sexual; do
  d="$BASE/baselines_v2/$ds"
  f="$d/results_qwen3_vl_nudity.txt"
  [ -f "$f" ] && echo "$ds: $(grep 'SR ' "$f")"
done

echo ""
echo "--- SAFREE ---"
for ds in rab mma p4dn unlearndiff mja_sexual; do
  d="$BASE/safree_reproduction/$ds/all"
  [ ! -d "$d" ] && d="$BASE/safree_reproduction/$ds"
  f="$d/results_qwen3_vl_nudity.txt"
  [ -f "$f" ] && echo "$ds: $(grep 'SR ' "$f")"
done

echo ""
echo "--- Ours ---"
for key in rab_best rab_fam rab_both mma_best p4dn_best udiff_best mja_sex_best; do
  d="${OURS_DIRS[$key]}"
  f="$d/results_qwen3_vl_nudity.txt"
  [ -f "$f" ] && echo "$key: $(grep 'SR ' "$f")"
done
