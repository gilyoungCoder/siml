#!/bin/bash
# ============================================================
# Full Evaluation: NudeNet + Qwen for all configs (priority order)
# Runs on specified GPUs, handles NudeNet → Qwen chain per config
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../../scripts/lib/repo_env.sh"

export PYTHONNOUSERSITE=1
PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
EVAL_NN="${UNLEARNING_REPO_ROOT}/vlm/eval_nudenet.py"
EVAL_VLM="${UNLEARNING_REPO_ROOT}/vlm/opensource_vlm_i2p_all.py"
BASE="${UNLEARNING_REPO_ROOT}/CAS_SpatialCFG/outputs"

cd /tmp

# ============================================================
# Helper: run NudeNet + Qwen eval on a directory
# ============================================================
eval_full() {
    local GPU=$1
    local DIR=$2
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then
        echo "[GPU $GPU] SKIP $NAME (dir not found)"
        return
    fi

    local NIMGS=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
    if [ "$NIMGS" -lt 10 ]; then
        echo "[GPU $GPU] SKIP $NAME (only $NIMGS images)"
        return
    fi

    # NudeNet
    if [ -f "$DIR/results_nudenet.txt" ]; then
        echo "[GPU $GPU] SKIP NN $NAME (exists)"
    else
        echo "[GPU $GPU] NN: $NAME ($NIMGS imgs)"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$DIR" 2>&1 | tail -3
    fi

    # Qwen
    if unlearning_find_qwen_result_txt "$DIR" >/dev/null 2>&1; then
        echo "[GPU $GPU] SKIP Qwen $NAME (exists)"
    else
        echo "[GPU $GPU] Qwen: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_VLM" "$DIR" nudity qwen 2>&1 | tail -5
    fi

    echo "[GPU $GPU] DONE: $NAME"
    echo ""
}

# ============================================================
# Wait for GPU memory to be below threshold
# ============================================================
wait_gpu_free() {
    local GPU=$1
    local MAX_MEM=${2:-1000}
    while true; do
        local MEM=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ "$MEM" -lt "$MAX_MEM" ] 2>/dev/null; then
            break
        fi
        echo "  Waiting GPU $GPU (${MEM}MiB)..."
        sleep 30
    done
}

# ============================================================
# Priority configs per GPU (sorted by NudeNet, best first)
# ============================================================

# GPU 0: v7 top configs (NudeNet <= 6%)
(
    echo "=== GPU 0: v7 Top Configs ==="
    G=0
    wait_gpu_free $G

    eval_full $G "$BASE/v7/v7_hyb_ts15_as15"
    eval_full $G "$BASE/v7/v7_hyb_ts20_as20"
    eval_full $G "$BASE/v7/v7_hyb_ts25_as25"
    eval_full $G "$BASE/v7/v7_hyb_ts30_as20"
    eval_full $G "$BASE/v7/v7_sld_s25"
    eval_full $G "$BASE/v7/v7_hyb_ts25_as15"
    eval_full $G "$BASE/v7/v7_hyb_ts20_as15"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as10"
    eval_full $G "$BASE/v7/v7_hyb_ts20_as10"
    eval_full $G "$BASE/v7/v7_sld_s20"

    echo "=== GPU 0: DONE ==="
) > "$BASE/eval_gpu0.log" 2>&1 &
PID0=$!

# GPU 2: v7 cas/spatial variants + v10
(
    echo "=== GPU 2: v7 Variants + v10 ==="
    G=2
    wait_gpu_free $G

    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_cas04"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_cas05"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_sig20"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_sig5"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_spat02"
    eval_full $G "$BASE/v10/v10_proj_ts2_as1"
    eval_full $G "$BASE/v10/v10_hfid_ts15_as15_d03"
    eval_full $G "$BASE/v10/v10_proj_ts3_as1_acas"

    echo "=== GPU 2: DONE ==="
) > "$BASE/eval_gpu2.log" 2>&1 &
PID2=$!

# GPU 3: v7 mid-tier + v11/v12 + COCO
(
    echo "=== GPU 3: v7 Mid + v11/v12 + COCO ==="
    G=3
    wait_gpu_free $G

    eval_full $G "$BASE/v11/v11_proj_K4_eta03"
    eval_full $G "$BASE/v12/v12_xattn_proj_ts2_as1"
    eval_full $G "$BASE/v7/v7_hyb_ts10_as15"
    eval_full $G "$BASE/v7/v7_hyb_ts10_as10"
    eval_full $G "$BASE/v7/v7_hyb_ts10_as20"
    eval_full $G "$BASE/v7/v7_sld_s15"
    eval_full $G "$BASE/v7/v7_ainp_s10"
    eval_full $G "$BASE/v7/COCO_v7_hyb_ts15_as15"
    eval_full $G "$BASE/v7/COCO_v7_hyb_ts10_as15"

    echo "=== GPU 3: DONE ==="
) > "$BASE/eval_gpu3.log" 2>&1 &
PID3=$!

# GPU 7: v6 + remaining v7
(
    echo "=== GPU 7: v6 + v7 Remaining ==="
    G=7
    wait_gpu_free $G

    eval_full $G "$BASE/v6/v6_crossattn_ts15_as15"
    eval_full $G "$BASE/v6/v6_crossattn_ts20_as15"
    eval_full $G "$BASE/v6/v6_crossattn_ts20_as20"
    eval_full $G "$BASE/v6/COCO_v6_crossattn"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_spat04"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_spat05"
    eval_full $G "$BASE/v7/v7_hyb_ts15_as15_cas07"
    eval_full $G "$BASE/v7/v7_proj_s1"
    eval_full $G "$BASE/v7/v7_sld_s10"
    eval_full $G "$BASE/v7/COCO_v7_hyb_ts20_as20"

    echo "=== GPU 7: DONE ==="
) > "$BASE/eval_gpu7.log" 2>&1 &
PID7=$!

echo "============================================"
echo "Full Eval Launched: $(date)"
echo "  GPU 0: v7 Top (PID $PID0)"
echo "  GPU 2: v7 Variants + v10 (PID $PID2)"
echo "  GPU 3: v11/v12 + v7 Mid (PID $PID3)"
echo "  GPU 7: v6 + v7 Rest (PID $PID7)"
echo "============================================"
echo "Monitor: tail -f $BASE/eval_gpu*.log"

wait $PID0 $PID2 $PID3 $PID7

echo ""
echo "============================================"
echo "ALL EVAL COMPLETE: $(date)"
echo "============================================"

# Final summary
echo ""
printf "%-40s | %8s | %8s\n" "Config" "NudeNet%" "Qwen SR%"
printf "%s\n" "$(printf '%.0s-' {1..62})"

for ver in v6 v7 v10 v11 v12; do
    for d in "$BASE/$ver"/*/; do
        [ -d "$d" ] || continue
        name="$ver/$(basename "$d")"

        nn="$(unlearning_nudenet_percent "$d" || echo -)"

        sr_raw="$(unlearning_qwen_percent_value "$d" SR || echo -)"
        sr="${sr_raw}%"
        [ "$sr_raw" = "-" ] && sr="-"

        printf "%-40s | %8s | %8s\n" "$name" "$nn" "$sr"
    done
done
