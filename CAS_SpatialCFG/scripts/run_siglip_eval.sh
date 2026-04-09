#!/bin/bash
# ============================================================
# SigLIP2 Safety Eval: All configs (v3-v12) on GPU 2,3
# Binary (Safe/Unsafe) + 5-class (Normal/Hentai/Porn/Enticing/Anime)
# ============================================================
set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
EVAL="/mnt/home3/yhgil99/unlearning/vlm/eval_siglip_safety.py"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"

cd /tmp

eval_dir() {
    local GPU=$1
    local DIR=$2
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then return; fi
    if [ -f "$DIR/results_siglip_safety.json" ]; then
        echo "[GPU $GPU] SKIP $NAME (done)"
        return
    fi
    local N=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
    if [ "$N" -lt 10 ]; then return; fi

    echo "[GPU $GPU] SigLIP: $NAME ($N imgs)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL" "$DIR" --mode both 2>&1 | tail -8
    echo ""
}

# GPU 2: v3 baseline + v4/v5 best + v6 + v7 top configs
(
    echo "=== GPU 2: SigLIP v3-v7 ==="

    # Baselines
    eval_dir 2 "$BASE/v3/baseline"
    eval_dir 2 "$BASE/v4/baseline"
    eval_dir 2 "$BASE/v5/baseline"

    # v3 best
    eval_dir 2 "$BASE/v3/dag_s5"
    eval_dir 2 "$BASE/v3/dag_s3"

    # v4 best
    eval_dir 2 "$BASE/v4/sld_s10"
    eval_dir 2 "$BASE/v4/ainp_s1.0_t0.1"

    # v5 best
    eval_dir 2 "$BASE/v5/sld_s10"

    # v6 best
    eval_dir 2 "$BASE/v6/v6_crossattn_ts15_as15"
    eval_dir 2 "$BASE/v6/v6_crossattn_ts20_as15"
    eval_dir 2 "$BASE/v6/v6_crossattn_ts20_as20"
    eval_dir 2 "$BASE/v6/v6_crossattn_ts10_as10"

    # v7 top (NudeNet < 7%)
    eval_dir 2 "$BASE/v7/v7_hyb_ts15_as15"
    eval_dir 2 "$BASE/v7/v7_hyb_ts15_as15_cas04"
    eval_dir 2 "$BASE/v7/v7_hyb_ts15_as15_cas05"
    eval_dir 2 "$BASE/v7/v7_hyb_ts20_as20"
    eval_dir 2 "$BASE/v7/v7_hyb_ts25_as25"
    eval_dir 2 "$BASE/v7/v7_hyb_ts30_as20"
    eval_dir 2 "$BASE/v7/v7_sld_s25"
    eval_dir 2 "$BASE/v7/v7_sld_s20"
    eval_dir 2 "$BASE/v7/v7_sld_s15"
    eval_dir 2 "$BASE/v7/v7_hyb_ts15_as10"
    eval_dir 2 "$BASE/v7/v7_hyb_ts20_as15"
    eval_dir 2 "$BASE/v7/v7_hyb_ts25_as15"
    eval_dir 2 "$BASE/v7/v7_hyb_ts20_as10"
    eval_dir 2 "$BASE/v7/v7_ainp_s10"

    # v7 mid-tier for completeness
    eval_dir 2 "$BASE/v7/v7_hyb_ts10_as15"
    eval_dir 2 "$BASE/v7/v7_hyb_ts10_as10"
    eval_dir 2 "$BASE/v7/v7_hyb_ts10_as20"
    eval_dir 2 "$BASE/v7/v7_sld_s10"
    eval_dir 2 "$BASE/v7/v7_proj_s1"

    # COCO
    eval_dir 2 "$BASE/v6/COCO_v6_crossattn"
    eval_dir 2 "$BASE/v7/COCO_v7_hyb_ts15_as15"
    eval_dir 2 "$BASE/v7/COCO_v7_hyb_ts20_as20"

    echo "=== GPU 2: DONE ==="
) > "$BASE/siglip_gpu2.log" 2>&1 &
PID2=$!

# GPU 3: v10/v11/v12 + v7 variants + remaining
(
    echo "=== GPU 3: SigLIP v10-v12 + v7 rest ==="

    # v10
    eval_dir 3 "$BASE/v10/v10_proj_ts2_as1"
    eval_dir 3 "$BASE/v10/v10_hfid_ts15_as15_d03"
    eval_dir 3 "$BASE/v10/v10_proj_ts3_as1_acas"

    # v11
    eval_dir 3 "$BASE/v11/v11_proj_K4_eta03"

    # v12
    eval_dir 3 "$BASE/v12/v12_xattn_proj_ts2_as1"

    # v7 variants (sig, spat, start, etc.)
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_sig20"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_sig5"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_spat02"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_spat04"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_spat05"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_cas07"
    eval_dir 3 "$BASE/v7/v7_hyb_ts15_as15_start02"
    eval_dir 3 "$BASE/v7/v7_hybproj_ts10_as15"
    eval_dir 3 "$BASE/v7/v7_sld_s5"
    eval_dir 3 "$BASE/v7/v7_hyb_ts5_as15"
    eval_dir 3 "$BASE/v7/v7_ainp_s07"

    # v6 remaining
    eval_dir 3 "$BASE/v6/v6_crossattn_ainp_s1"
    eval_dir 3 "$BASE/v6/v6_crossattn_ainp_s07"
    eval_dir 3 "$BASE/v6/v6_crossattn_proj_s1"
    eval_dir 3 "$BASE/v6/v6_crossattn_ts10_as15"

    echo "=== GPU 3: DONE ==="
) > "$BASE/siglip_gpu3.log" 2>&1 &
PID3=$!

echo "SigLIP eval launched: GPU 2 (PID $PID2), GPU 3 (PID $PID3)"
echo "Monitor: tail -f $BASE/siglip_gpu*.log"

wait $PID2 $PID3

echo ""
echo "============================================"
echo "ALL SIGLIP EVAL COMPLETE: $(date)"
echo "============================================"

# Summary
echo ""
printf "%-35s | %8s | %8s | %8s | %8s\n" "Config" "SigBin%" "Sig5c%" "NudeNet%" "QwenSR%"
printf "%s\n" "$(printf '%.0s-' {1..85})"

for ver in v3 v6 v7 v10 v11 v12; do
    for d in "$BASE/$ver"/*/; do
        [ -d "$d" ] || continue
        name="$ver/$(basename $d)"
        [[ "$name" == *debug* ]] && continue

        sig_bin="-"; sig_5c="-"; nn="-"; sr="-"

        if [ -f "$d/results_siglip_safety.json" ]; then
            sig_bin=$(python3.10 -c "import json; d=json.load(open('$d/results_siglip_safety.json')); print(f'{d[\"binary\"][\"unsafe_rate\"]*100:.1f}')" 2>/dev/null || echo "-")
            sig_5c=$(python3.10 -c "import json; d=json.load(open('$d/results_siglip_safety.json')); print(f'{d[\"5class\"][\"unsafe_rate\"]*100:.1f}')" 2>/dev/null || echo "-")
        fi

        [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP '\d+\.\d+%' "$d/results_nudenet.txt" | head -1)

        qfile=""
        [ -f "$d/results_qwen3_vl_nudity.txt" ] && qfile="$d/results_qwen3_vl_nudity.txt"
        [ -f "$d/results_qwen_nudity.txt" ] && qfile="$d/results_qwen_nudity.txt"
        if [ -n "$qfile" ]; then
            s=$(grep -oP 'Safe:\s*\K\d+' "$qfile" 2>/dev/null || echo 0)
            p=$(grep -oP 'Partial:\s*\K\d+' "$qfile" 2>/dev/null || echo 0)
            nr=$(grep -oP 'NotRel:\s*\K\d+' "$qfile" 2>/dev/null || echo 0)
            f=$(grep -oP 'Full:\s*\K\d+' "$qfile" 2>/dev/null || echo 0)
            t=$((s+p+nr+f))
            [ "$t" -gt 0 ] && sr="$(echo "scale=1; ($s+$p)*100/$t" | bc)%"
        fi

        [ "$sig_bin" != "-" ] || [ "$nn" != "-" ] && \
        printf "%-35s | %8s | %8s | %8s | %8s\n" "$name" "${sig_bin}%" "${sig_5c}%" "$nn" "$sr"
    done
done
