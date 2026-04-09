#!/bin/bash
# ============================================================
# VQAScore Evaluation: All versions (V4+) + country_nude_body generation
# Step 1: Generate country_nude_body images for each version's best config
# Step 2: Run VQAScore on Ring-A-Bell outputs (original prompt alignment)
# Step 3: Run VQAScore alignment on anchor subset
# ============================================================
set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
EVAL_VQA="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore.py"
EVAL_VQA_ALIGN="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore_alignment.py"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"

RINGABELL="$BASE/prompts/nudity-ring-a-bell.csv"
RINGABELL_TXT="$BASE/prompts/ringabell.txt"
ANCHOR_SUBSET="$BASE/prompts/ringabell_anchor_subset.csv"
COUNTRY_NUDE="$BASE/prompts/country_nude_body.txt"
COUNTRY_CSV="$BASE/prompts/country_nude_body.csv"
CONCEPT_DIR="$BASE/exemplars/sd14/concept_directions.pt"

cd /tmp

# ============================================================
# Helper: wait for GPU
# ============================================================
wait_gpu_free() {
    local GPU=$1
    while true; do
        local MEM=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ "$MEM" -lt 1000 ] 2>/dev/null; then break; fi
        echo "  Waiting GPU $GPU (${MEM}MiB)..."
        sleep 30
    done
}

# ============================================================
# Helper: run VQAScore on a directory (skip if done)
# ============================================================
run_vqa() {
    local GPU=$1
    local DIR=$2
    local PROMPT_FILE=$3
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then echo "[GPU $GPU] SKIP $NAME (no dir)"; return; fi
    if [ -f "$DIR/results_vqascore.json" ]; then echo "[GPU $GPU] SKIP VQA $NAME (done)"; return; fi

    local NIMGS=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -lt 10 ]; then echo "[GPU $GPU] SKIP $NAME ($NIMGS imgs)"; return; fi

    echo "[GPU $GPU] VQA: $NAME ($NIMGS imgs)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_VQA" "$DIR" --prompts "$PROMPT_FILE" 2>&1 | tail -5
    echo "[GPU $GPU] VQA DONE: $NAME"
}

# ============================================================
# Helper: run VQAScore alignment (anchor/erased)
# ============================================================
run_vqa_align() {
    local GPU=$1
    local DIR=$2
    local PROMPT_FILE=$3
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then return; fi
    if [ -f "$DIR/results_vqascore_alignment.json" ]; then echo "[GPU $GPU] SKIP ALIGN $NAME (done)"; return; fi

    local NIMGS=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -lt 10 ]; then return; fi

    echo "[GPU $GPU] ALIGN: $NAME"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_VQA_ALIGN" "$DIR" --prompts "$PROMPT_FILE" --prompt_type all 2>&1 | tail -8
}

# ============================================================
# Helper: generate + nudenet for country_nude_body
# ============================================================
gen_country() {
    local GPU=$1
    local GEN_SCRIPT=$2
    local OUTDIR=$3
    shift 3
    local ARGS="$@"
    local NAME=$(basename "$OUTDIR")
    local MIN_IMGS=80  # 20 prompts * 4 samples

    local NIMGS=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -ge "$MIN_IMGS" ]; then
        echo "[GPU $GPU] SKIP gen $NAME ($NIMGS imgs)"
    else
        echo "[GPU $GPU] GEN country: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GEN_SCRIPT" \
            --prompts "$COUNTRY_NUDE" --outdir "$OUTDIR" $ARGS 2>&1 | tail -3
    fi

    # NudeNet
    if [ ! -f "$OUTDIR/results_nudenet.txt" ] && [ -d "$OUTDIR" ]; then
        echo "[GPU $GPU] NN: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$OUTDIR" 2>&1 | tail -3
    fi
}

# ============================================================
# PHASE 1: Generate country_nude_body images (4 GPUs parallel)
# Best configs per version, split across GPUs
# ============================================================

# GPU 0: baseline + v4 best + v5 best
(
    echo "=== GPU 0: Country Gen — baseline + v4 + v5 ==="
    G=0
    wait_gpu_free $G

    # Baseline (no guidance)
    gen_country $G "$BASE/generate_baseline.py" \
        "$BASE/outputs/country/baseline"

    # V4 best: sld_s10
    gen_country $G "$BASE/generate_v4.py" \
        "$BASE/outputs/country/v4_sld_s10" \
        --guide_mode sld --safety_scale 10.0 --cas_threshold 0.6

    # V5 best: sld_s10
    gen_country $G "$BASE/generate_v5.py" \
        "$BASE/outputs/country/v5_sld_s10" \
        --guide_mode sld --safety_scale 10.0 --cas_threshold 0.6

    echo "=== GPU 0: Country Gen DONE ==="
) > "$BASE/outputs/vqa_gpu0.log" 2>&1 &
PID0=$!

# GPU 5: v6 best + v7 best
(
    echo "=== GPU 5: Country Gen — v6 + v7 ==="
    G=5
    wait_gpu_free $G

    # V6 best: crossattn ts20_as15
    gen_country $G "$BASE/generate_v6.py" \
        "$BASE/outputs/country/v6_ts20_as15" \
        --guide_mode hybrid --target_scale 20 --anchor_scale 15 --cas_threshold 0.6

    # V7 best: hyb_ts15_as15
    gen_country $G "$BASE/generate_v7.py" \
        "$BASE/outputs/country/v7_hyb_ts15_as15" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 15 --anchor_scale 15 --cas_threshold 0.6

    # V7 runner-up: hyb_ts25_as25
    gen_country $G "$BASE/generate_v7.py" \
        "$BASE/outputs/country/v7_hyb_ts25_as25" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode hybrid --target_scale 25 --anchor_scale 25 --cas_threshold 0.6

    echo "=== GPU 5: Country Gen DONE ==="
) > "$BASE/outputs/vqa_gpu5.log" 2>&1 &
PID5=$!

# GPU 7: v10 + v11 + v12
(
    echo "=== GPU 7: Country Gen — v10 + v11 + v12 ==="
    G=7
    wait_gpu_free $G

    # V10 best: proj_ts2_as1
    gen_country $G "$BASE/generate_v10.py" \
        "$BASE/outputs/country/v10_proj_ts2_as1" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode proj_anchor --target_scale 2 --anchor_scale 1 --cas_threshold 0.6

    # V11: proj_K4_eta03
    gen_country $G "$BASE/generate_v11.py" \
        "$BASE/outputs/country/v11_proj_K4_eta03" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode proj_anchor --target_scale 2 --anchor_scale 1 \
        --K_ensemble 4 --eta 0.3 --cas_threshold 0.6

    # V12: xattn_proj_ts2_as1
    gen_country $G "$BASE/generate_v12.py" \
        "$BASE/outputs/country/v12_xattn_proj_ts2_as1" \
        --exemplar_mode exemplar --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode proj_anchor --target_scale 2 --anchor_scale 1 --cas_threshold 0.6

    echo "=== GPU 7: Country Gen DONE ==="
) > "$BASE/outputs/vqa_gpu7.log" 2>&1 &
PID7=$!

echo "Phase 1 (Country Gen) launched: PIDs $PID0 $PID5 $PID7"
wait $PID0 $PID5 $PID7
echo "Phase 1 DONE: $(date)"

# ============================================================
# PHASE 2: VQAScore on Ring-A-Bell best configs + country outputs
# ============================================================

# GPU 0: VQAScore on Ring-A-Bell (v3/v4/v5 best + baseline)
(
    echo "=== GPU 0: VQA Ring-A-Bell v3/v4/v5 ==="
    G=0
    wait_gpu_free $G

    run_vqa $G "$BASE/outputs/v3/baseline" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v3/dag_s5" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v3/dag_s3" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v3/sld_s3_cas05" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v4/baseline" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v4/sld_s10" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v4/ainp_s1.0_t0.1" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v5/baseline" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v5/sld_s10" "$RINGABELL_TXT"

    echo "=== GPU 0: VQA DONE ==="
) > "$BASE/outputs/vqa_phase2_gpu0.log" 2>&1 &
PID0=$!

# GPU 5: VQAScore on Ring-A-Bell (v6/v7 best configs) + country
(
    echo "=== GPU 5: VQA Ring-A-Bell v6/v7 + country ==="
    G=5
    wait_gpu_free $G

    run_vqa $G "$BASE/outputs/v6/v6_crossattn_ts15_as15" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v6/v6_crossattn_ts20_as15" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v6/v6_crossattn_ts20_as20" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_hyb_ts15_as15" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_hyb_ts20_as20" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_hyb_ts25_as25" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_hyb_ts30_as20" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_sld_s25" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_ainp_s10" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v7/v7_hyb_ts15_as15_cas04" "$RINGABELL_TXT"

    # VQA on country outputs
    for d in "$BASE/outputs/country"/*/; do
        [ -d "$d" ] && run_vqa $G "$d" "$COUNTRY_NUDE"
    done

    echo "=== GPU 5: VQA DONE ==="
) > "$BASE/outputs/vqa_phase2_gpu5.log" 2>&1 &
PID5=$!

# GPU 7: VQAScore on Ring-A-Bell (v10/v11/v12) + anchor alignment
(
    echo "=== GPU 7: VQA Ring-A-Bell v10/v11/v12 + Anchor Align ==="
    G=7
    wait_gpu_free $G

    run_vqa $G "$BASE/outputs/v10/v10_proj_ts2_as1" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v10/v10_hfid_ts15_as15_d03" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v10/v10_proj_ts3_as1_acas" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v11/v11_proj_K4_eta03" "$RINGABELL_TXT"
    run_vqa $G "$BASE/outputs/v12/v12_xattn_proj_ts2_as1" "$RINGABELL_TXT"

    # Anchor alignment on best configs
    for d in \
        "$BASE/outputs/v3/baseline" \
        "$BASE/outputs/v3/dag_s5" \
        "$BASE/outputs/v7/v7_hyb_ts15_as15" \
        "$BASE/outputs/v7/v7_hyb_ts25_as25" \
        "$BASE/outputs/v10/v10_proj_ts2_as1" \
        "$BASE/outputs/v6/v6_crossattn_ts20_as15"; do
        run_vqa_align $G "$d" "$ANCHOR_SUBSET"
    done

    echo "=== GPU 7: VQA DONE ==="
) > "$BASE/outputs/vqa_phase2_gpu7.log" 2>&1 &
PID7=$!

echo "Phase 2 (VQAScore) launched: PIDs $PID0 $PID5 $PID7"
wait $PID0 $PID5 $PID7
echo "Phase 2 DONE: $(date)"

# ============================================================
# PHASE 3: VQAScore on country_nude_body with anchor prompts
# ============================================================
echo "=== Phase 3: Country VQA Alignment ==="
G=0
for d in "$BASE/outputs/country"/*/; do
    [ -d "$d" ] || continue
    NAME=$(basename "$d")
    if [ -f "$d/results_vqascore_alignment.json" ]; then
        echo "SKIP ALIGN $NAME"
        continue
    fi
    echo "ALIGN: $NAME"
    CUDA_VISIBLE_DEVICES=$G $PYTHON "$EVAL_VQA_ALIGN" "$d" \
        --prompts "$COUNTRY_CSV" --prompt_type all 2>&1 | tail -5
done

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "============================================"
echo "FINAL VQA RESULTS SUMMARY: $(date)"
echo "============================================"
printf "%-40s | %8s | %8s\n" "Config" "VQA" "NN%"
printf "%s\n" "$(printf '%.0s-' {1..62})"

for d in "$BASE/outputs/v3/baseline" \
         "$BASE/outputs/v3/dag_s5" \
         "$BASE/outputs/v4/sld_s10" \
         "$BASE/outputs/v5/sld_s10" \
         "$BASE/outputs/v6/v6_crossattn_ts15_as15" \
         "$BASE/outputs/v6/v6_crossattn_ts20_as15" \
         "$BASE/outputs/v7/v7_hyb_ts15_as15" \
         "$BASE/outputs/v7/v7_hyb_ts25_as25" \
         "$BASE/outputs/v7/v7_hyb_ts30_as20" \
         "$BASE/outputs/v10/v10_proj_ts2_as1" \
         "$BASE/outputs/v11/v11_proj_K4_eta03" \
         "$BASE/outputs/v12/v12_xattn_proj_ts2_as1"; do
    [ -d "$d" ] || continue
    name=$(basename "$d")

    vqa="-"
    [ -f "$d/results_vqascore.json" ] && vqa=$(python3 -c "import json; d=json.load(open('$d/results_vqascore.json')); print(f'{d[\"mean\"]:.4f}')" 2>/dev/null || echo "-")

    nn="-"
    [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP '\d+\.\d+%' "$d/results_nudenet.txt" | head -1)

    printf "%-40s | %8s | %8s\n" "$name" "$vqa" "$nn"
done

echo ""
echo "=== Country Nude Body Results ==="
for d in "$BASE/outputs/country"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    vqa="-"; nn="-"; anchor="-"
    [ -f "$d/results_vqascore.json" ] && vqa=$(python3 -c "import json; d=json.load(open('$d/results_vqascore.json')); print(f'{d[\"mean\"]:.4f}')" 2>/dev/null || echo "-")
    [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP '\d+\.\d+%' "$d/results_nudenet.txt" | head -1)
    [ -f "$d/results_vqascore_alignment.json" ] && anchor=$(python3 -c "import json; d=json.load(open('$d/results_vqascore_alignment.json')); print(f'{d[\"summary\"][\"anchor\"][\"mean\"]:.4f}')" 2>/dev/null || echo "-")
    printf "%-25s | VQA=%s | NN=%s | Anchor=%s\n" "$name" "$vqa" "$nn" "$anchor"
done

echo ""
echo "ALL DONE: $(date)"
