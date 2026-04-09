#!/bin/bash
# ============================================================
# Remaining evaluations — 8 GPU parallel
# SigLIP country, VQA alignment, Qwen country, VQA extra
# ============================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
EVAL_VQA="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore.py"
EVAL_VQA_ALIGN="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore_alignment.py"
EVAL_SIG="/mnt/home3/yhgil99/unlearning/vlm/eval_siglip_safety.py"
RINGABELL_TXT="$BASE/../prompts/ringabell.txt"
ANCHOR_ALL="$BASE/../prompts/anchor_friendly_all.csv"
COUNTRY_CSV="$BASE/../prompts/country_nude_body.csv"
COUNTRY_TXT="$BASE/../prompts/country_nude_body.txt"

cd /tmp

# ============================================================
# GPU 0: Qwen eval on country (baseline, v4, v5) — heavy ~17GB
# ============================================================
(
    echo "=== GPU 0: Qwen country baseline/v4/v5 ==="
    for d in baseline v4_sld_s10 v5_sld_s10; do
        dir="$BASE/country/$d"
        [ -d "$dir" ] || continue
        [ -f "$dir/results_qwen_nudity.txt" ] || [ -f "$dir/results_qwen3_vl_nudity.txt" ] && continue
        echo "[GPU 0] Qwen: $d"
        CUDA_VISIBLE_DEVICES=0 $PYTHON "$EVAL_VLM" "$dir" nudity qwen 2>&1 | tail -3
    done
    echo "=== GPU 0: DONE ==="
) > "$BASE/remaining_gpu0.log" 2>&1 &

# ============================================================
# GPU 1: Qwen eval on country (v6, v7_ts15, v7_ts25)
# ============================================================
(
    echo "=== GPU 1: Qwen country v6/v7 ==="
    for d in v6_ts20_as15 v7_hyb_ts15_as15 v7_hyb_ts25_as25; do
        dir="$BASE/country/$d"
        [ -d "$dir" ] || continue
        [ -f "$dir/results_qwen_nudity.txt" ] || [ -f "$dir/results_qwen3_vl_nudity.txt" ] && continue
        echo "[GPU 1] Qwen: $d"
        CUDA_VISIBLE_DEVICES=1 $PYTHON "$EVAL_VLM" "$dir" nudity qwen 2>&1 | tail -3
    done
    echo "=== GPU 1: DONE ==="
) > "$BASE/remaining_gpu1.log" 2>&1 &

# ============================================================
# GPU 2: Qwen eval on country (v10, v11, v12)
# ============================================================
(
    echo "=== GPU 2: Qwen country v10/v11/v12 ==="
    for d in v10_proj_ts2_as1 v11_proj_K4_eta03 v12_xattn_proj_ts2_as1; do
        dir="$BASE/country/$d"
        [ -d "$dir" ] || continue
        [ -f "$dir/results_qwen_nudity.txt" ] || [ -f "$dir/results_qwen3_vl_nudity.txt" ] && continue
        echo "[GPU 2] Qwen: $d"
        CUDA_VISIBLE_DEVICES=2 $PYTHON "$EVAL_VLM" "$dir" nudity qwen 2>&1 | tail -3
    done
    echo "=== GPU 2: DONE ==="
) > "$BASE/remaining_gpu2.log" 2>&1 &

# ============================================================
# GPU 3: SigLIP on ALL country outputs
# ============================================================
(
    echo "=== GPU 3: SigLIP country ==="
    for d in "$BASE/country"/*/; do
        [ -d "$d" ] || continue
        [ -f "$d/results_siglip_safety.json" ] && continue
        echo "[GPU 3] SigLIP: $(basename $d)"
        CUDA_VISIBLE_DEVICES=3 $PYTHON "$EVAL_SIG" "$d" --mode both 2>&1 | tail -5
    done
    echo "=== GPU 3: DONE ==="
) > "$BASE/remaining_gpu3.log" 2>&1 &

# ============================================================
# GPU 4: VQA alignment on Ring-A-Bell top configs (anchor_friendly_all.csv)
# ============================================================
(
    echo "=== GPU 4: VQA Alignment (anchor_friendly_all) ==="
    for d in \
        "$BASE/v3/dag_s5" \
        "$BASE/v3/dag_s3" \
        "$BASE/v3/baseline" \
        "$BASE/v7/v7_hyb_ts15_as15" \
        "$BASE/v7/v7_hyb_ts25_as25" \
        "$BASE/v10/v10_proj_ts2_as1" \
        "$BASE/v6/v6_crossattn_ts20_as15" \
        "$BASE/v6/v6_crossattn_ts20_as20" \
        "$BASE/v4/sld_s10" \
        "$BASE/v5/sld_s10" \
        "$BASE/v12/v12_xattn_proj_ts2_as1" \
        "$BASE/v11/v11_proj_K4_eta03" \
        "$BASE/v4/ainp_s1.0_t0.1" \
        "$BASE/v7/v7_hyb_ts10_as20" \
        "$BASE/v4/baseline"; do
        [ -d "$d" ] || continue
        [ -f "$d/results_vqascore_alignment.json" ] && { echo "SKIP $(basename $d)"; continue; }
        echo "[GPU 4] ALIGN: $(basename $d)"
        CUDA_VISIBLE_DEVICES=4 $PYTHON "$EVAL_VQA_ALIGN" "$d" --prompts "$ANCHOR_ALL" --prompt_type all 2>&1 | tail -5
    done
    echo "=== GPU 4: DONE ==="
) > "$BASE/remaining_gpu4.log" 2>&1 &

# ============================================================
# GPU 5: VQAScore on extra configs (cas04, cas05, sld_s25, ainp_s10, etc.)
# ============================================================
(
    echo "=== GPU 5: VQA extra configs ==="
    for d in \
        "$BASE/v7/v7_hyb_ts15_as15_cas04" \
        "$BASE/v7/v7_hyb_ts15_as15_cas05" \
        "$BASE/v7/v7_sld_s25" \
        "$BASE/v7/v7_ainp_s10" \
        "$BASE/v7/v7_hyb_ts10_as20" \
        "$BASE/v7/v7_hyb_ts10_as15" \
        "$BASE/v7/v7_sld_s15" \
        "$BASE/v6/v6_crossattn_sld_s15" \
        "$BASE/v6/v6_crossattn_sld_s10" \
        "$BASE/v6/v6_crossattn_ts10_as10"; do
        [ -d "$d" ] || continue
        [ -f "$d/results_vqascore.json" ] && { echo "SKIP $(basename $d)"; continue; }
        N=$(ls "$d"/*.png 2>/dev/null | wc -l)
        [ "$N" -lt 10 ] && continue
        echo "[GPU 5] VQA: $(basename $d)"
        CUDA_VISIBLE_DEVICES=5 $PYTHON "$EVAL_VQA" "$d" --prompts "$RINGABELL_TXT" 2>&1 | tail -3
    done
    echo "=== GPU 5: DONE ==="
) > "$BASE/remaining_gpu5.log" 2>&1 &

# ============================================================
# GPU 6: VQAScore on v4/v5 top configs
# ============================================================
(
    echo "=== GPU 6: VQA v4/v5 configs ==="
    for d in \
        "$BASE/v4/ainp_s1.0_t0.1" \
        "$BASE/v4/sld_s7" \
        "$BASE/v4/ainp_s1.0_t0.2" \
        "$BASE/v5/ainp_s1.0_p-0.1" \
        "$BASE/v5/sld_s7" \
        "$BASE/v3/dag_s3" \
        "$BASE/v3/sld_s3_cas05" \
        "$BASE/v6/v6_crossattn_ts20_as20"; do
        [ -d "$d" ] || continue
        [ -f "$d/results_vqascore.json" ] && { echo "SKIP $(basename $d)"; continue; }
        N=$(ls "$d"/*.png 2>/dev/null | wc -l)
        [ "$N" -lt 10 ] && continue
        echo "[GPU 6] VQA: $(basename $d)"
        CUDA_VISIBLE_DEVICES=6 $PYTHON "$EVAL_VQA" "$d" --prompts "$RINGABELL_TXT" 2>&1 | tail -3
    done
    echo "=== GPU 6: DONE ==="
) > "$BASE/remaining_gpu6.log" 2>&1 &

# ============================================================
# GPU 7: VQA alignment on country outputs
# ============================================================
(
    echo "=== GPU 7: VQA Alignment country ==="
    for d in "$BASE/country"/*/; do
        [ -d "$d" ] || continue
        [ -f "$d/results_vqascore_alignment.json" ] && { echo "SKIP $(basename $d)"; continue; }
        echo "[GPU 7] ALIGN country: $(basename $d)"
        CUDA_VISIBLE_DEVICES=7 $PYTHON "$EVAL_VQA_ALIGN" "$d" --prompts "$COUNTRY_CSV" --prompt_type all 2>&1 | tail -5
    done
    echo "=== GPU 7: DONE ==="
) > "$BASE/remaining_gpu7.log" 2>&1 &

echo "============================================"
echo "Remaining eval launched on ALL 8 GPUs: $(date)"
echo "  GPU 0: Qwen country (baseline/v4/v5)"
echo "  GPU 1: Qwen country (v6/v7)"
echo "  GPU 2: Qwen country (v10/v11/v12)"
echo "  GPU 3: SigLIP country"
echo "  GPU 4: VQA alignment (Ring-A-Bell anchor_friendly)"
echo "  GPU 5: VQA extra configs"
echo "  GPU 6: VQA v4/v5 configs"
echo "  GPU 7: VQA alignment country"
echo "============================================"

wait

echo ""
echo "ALL REMAINING EVAL COMPLETE: $(date)"
