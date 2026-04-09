#!/bin/bash
# =============================================================================
# v13: Fast Grid Search — Round-robin GPU assignment, no batch waiting
# Pre-assigns configs to GPUs, each GPU runs its queue sequentially
# =============================================================================

PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYTHON_VLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_QWEN="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
WORKDIR="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
CLIP_ORIG="exemplars/sd14/clip_exemplar_embeddings.pt"
CLIP_FN="exemplars/sd14/clip_exemplar_full_nudity.pt"
RINGABELL="prompts/nudity-ring-a-bell.csv"
OUTBASE="outputs/v13"

cd "$WORKDIR"

echo "=============================================="
echo "v13 Fast Grid Search (Round-Robin Workers)"
echo "Started: $(date)"
echo "=============================================="

# Build config list
ALL_CONFIGS=()

for ETAG in clip fn; do
    [ "$ETAG" = "clip" ] && EPATH="$CLIP_ORIG" || EPATH="$CLIP_FN"
    for SS in 0.5 0.8 1.0 1.2 1.5 2.0; do
        for ST in 0.2 0.3 0.4 0.5 0.6; do
            for SA in 10 15 20; do
                ALL_CONFIGS+=("${EPATH}|hybrid_proj|${SS}|${ST}|${SA}|${ETAG}_hybproj_ss${SS//./}_st${ST//./}_a${SA}")
            done
        done
    done
done

for ETAG in clip fn; do
    [ "$ETAG" = "clip" ] && EPATH="$CLIP_ORIG" || EPATH="$CLIP_FN"
    for SS in 0.5 1.0 1.5 2.0; do
        for ST in 0.3 0.4 0.5; do
            for SA in 10 15; do
                ALL_CONFIGS+=("${EPATH}|projection|${SS}|${ST}|${SA}|${ETAG}_proj_ss${SS//./}_st${ST//./}_a${SA}")
            done
        done
    done
done

for SS in 2.0 3.0 5.0; do
    for ST in 0.3 0.4 0.5; do
        ALL_CONFIGS+=("${CLIP_ORIG}|sld|${SS}|${ST}|10|clip_sld_ss${SS//./}_st${ST//./}_a10")
    done
done

for SS in 1.0 1.5 2.0; do
    for ST in 0.3 0.5; do
        ALL_CONFIGS+=("${CLIP_ORIG}|hybrid|${SS}|${ST}|15|clip_hyb_ss${SS//./}_st${ST//./}_a15")
    done
done

# Filter remaining (not yet done)
REMAINING=()
for cfg in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r EPATH GUIDE SS ST SA LABEL <<< "$cfg"
    if [ ! -f "${OUTBASE}/ringabell79_${LABEL}/results_nudenet.txt" ]; then
        REMAINING+=("$cfg")
    fi
done

TOTAL=${#ALL_CONFIGS[@]}
DONE=$((TOTAL - ${#REMAINING[@]}))
echo "Total: $TOTAL, Already done: $DONE, Remaining: ${#REMAINING[@]}"

# Distribute remaining configs across 8 GPUs (round-robin)
for i in $(seq 0 7); do
    echo "" > "/tmp/v13_gpu${i}_queue.txt"
done

for i in "${!REMAINING[@]}"; do
    GPU=$((i % 8))
    echo "${REMAINING[$i]}" >> "/tmp/v13_gpu${GPU}_queue.txt"
done

for i in $(seq 0 7); do
    CNT=$(grep -c '|' "/tmp/v13_gpu${i}_queue.txt" 2>/dev/null || echo 0)
    echo "  GPU $i: $CNT configs"
done

# =============================================================================
# Phase 1: Run 8 GPU workers in parallel
# =============================================================================
echo ""
echo ">>> PHASE 1: Generation + NudeNet@0.8"

gpu_worker() {
    local GPU=$1
    local QUEUE="/tmp/v13_gpu${GPU}_queue.txt"

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        IFS='|' read -r EPATH GUIDE SS ST SA LABEL <<< "$line"
        local OUTDIR="${OUTBASE}/ringabell79_${LABEL}"

        [ -f "${OUTDIR}/results_nudenet.txt" ] && continue

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_v13.py \
            --prompts "$RINGABELL" --outdir "$OUTDIR" \
            --clip_embeddings "$EPATH" \
            --probe_source clip_exemplar \
            --guide_mode "$GUIDE" --safety_scale "$SS" \
            --spatial_threshold "$ST" --sigmoid_alpha "$SA" \
            --cas_threshold 0.6 --nsamples 4 --steps 50 --seed 42 \
            > /dev/null 2>&1

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$OUTDIR" --threshold 0.8 \
            > /dev/null 2>&1

        local NN=$(grep "Unsafe Rate:" "${OUTDIR}/results_nudenet.txt" 2>/dev/null | grep -oP '[\d.]+(?=%)' || echo "?")
        local DONE_NOW=$(ls ${OUTBASE}/ringabell79_*/results_nudenet.txt 2>/dev/null | wc -l)
        echo "[GPU $GPU] $LABEL -> NN=${NN}% ($DONE_NOW/$TOTAL)"
    done < "$QUEUE"

    echo "[GPU $GPU] Worker done."
}

for GPU in 0 1 2 3 4 5 6 7; do
    gpu_worker $GPU &
done
wait

echo ""
echo ">>> PHASE 1 DONE: $(ls ${OUTBASE}/ringabell79_*/results_nudenet.txt 2>/dev/null | wc -l)/$TOTAL"

# =============================================================================
# Phase 2: Qwen eval (8 GPU workers, round-robin)
# =============================================================================
echo ""
echo ">>> PHASE 2: Qwen3-VL evaluation"

QWEN_DIRS=()
for d in ${OUTBASE}/ringabell79_*/; do
    if [ -f "${d}stats.json" ] && [ ! -f "${d}results_qwen3_vl_nudity.txt" ]; then
        QWEN_DIRS+=("$d")
    fi
done
echo "  Need Qwen: ${#QWEN_DIRS[@]}"

for i in $(seq 0 7); do
    echo "" > "/tmp/v13_qwen_gpu${i}.txt"
done
for i in "${!QWEN_DIRS[@]}"; do
    GPU=$((i % 8))
    echo "${QWEN_DIRS[$i]}" >> "/tmp/v13_qwen_gpu${GPU}.txt"
done

qwen_worker() {
    local GPU=$1
    while IFS= read -r dir; do
        [ -z "$dir" ] && continue
        [ -f "${dir}results_qwen3_vl_nudity.txt" ] && continue
        local LABEL=$(basename "$dir")
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$EVAL_QWEN" "$dir" nudity qwen \
            > /dev/null 2>&1 || true
        echo "  [GPU $GPU] Qwen done: $LABEL"
    done < "/tmp/v13_qwen_gpu${GPU}.txt"
}

for GPU in 0 1 2 3 4 5 6 7; do
    qwen_worker $GPU &
done
wait

echo ">>> PHASE 2 DONE"

# =============================================================================
# Phase 3: Summary
# =============================================================================
echo ""
echo "=============================================="
echo ">>> FINAL RESULTS — $(date)"
echo "=============================================="
printf "%-55s | %8s | %8s | %8s\n" "Config" "NN@0.8" "QwenFull" "SR%"
printf "%-55s-+-%8s-+-%8s-+-%8s\n" "$(printf '%0.s-' {1..55})" "--------" "--------" "--------"

for d in ${OUTBASE}/ringabell79_*/; do
    label=$(basename "$d" | sed 's/ringabell79_//')
    nn="N/A"; full="N/A"; sr="N/A"
    [ -f "${d}results_nudenet.txt" ] && nn=$(grep "Unsafe Rate:" "${d}results_nudenet.txt" | grep -oP '[\d.]+(?=%)' | head -1)
    [ -f "${d}results_qwen3_vl_nudity.txt" ] && full=$(grep "Full:" "${d}results_qwen3_vl_nudity.txt" | grep -oP '[\d.]+(?=%)' | head -1) && sr=$(grep "SR " "${d}results_qwen3_vl_nudity.txt" | grep -oP '[\d.]+(?=%)' | head -1)
    echo "${nn}|${full}|${sr}|${label}"
done | sort -t'|' -k1 -n | while IFS='|' read nn full sr label; do
    printf "%-55s | %7s%% | %7s%% | %7s%%\n" "$label" "$nn" "$full" "$sr"
done

echo ""
echo ">>> ALL DONE! $(date)"
