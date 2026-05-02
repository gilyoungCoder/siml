#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate vlm

SAVE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"

echo "=== Remaining Qwen Eval START $(date) ==="

# Wait for SAFREE generation to finish (violence, shocking still running on siml-02)
echo "Waiting for SAFREE generation..."
while true; do
    v=$(find $SAVE/i2p_violence_v2 -name '*.png' 2>/dev/null | wc -l)
    s=$(find $SAVE/i2p_shocking_v2 -name '*.png' 2>/dev/null | wc -l)
    p=$(find $SAVE/p4dn -name '*.png' 2>/dev/null | wc -l)
    echo "  violence=$v shocking=$s p4dn=$p"
    [ "$v" -ge 700 ] && [ "$s" -ge 800 ] && [ "$p" -ge 150 ] && break
    sleep 60
done
echo "All ready!"

# SAFREE P4DN
echo "[Qwen] SAFREE p4dn"
CUDA_VISIBLE_DEVICES=0 python3 vlm/opensource_vlm_i2p_all.py "$SAVE/p4dn/all" nudity qwen 2>&1 | tail -2 &

# SAFREE Violence v2
echo "[Qwen] SAFREE violence"
CUDA_VISIBLE_DEVICES=1 python3 vlm/opensource_vlm_i2p_all.py "$SAVE/i2p_violence_v2" violence qwen 2>&1 | tail -2 &

# SAFREE Shocking v2
echo "[Qwen] SAFREE shocking"
CUDA_VISIBLE_DEVICES=2 python3 vlm/opensource_vlm_i2p_all.py "$SAVE/i2p_shocking_v2" shocking qwen 2>&1 | tail -2 &

# SAFREE Artists
for artist in vangogh picasso monet rembrandt warhol hopper; do
    d="$SAVE/artist_${artist}"
    # Merge seed subdirs into one for eval
    merged="$d/all_seeds"
    mkdir -p "$merged"
    cp $d/seed_*/*.png "$merged/" 2>/dev/null || true
    echo "[Qwen] SAFREE artist_$artist"
    CUDA_VISIBLE_DEVICES=3 python3 vlm/opensource_vlm_i2p_all.py "$merged" "style_${artist}" qwen 2>&1 | tail -2
done

wait

# Also run Qwen eval for MJA tuning results (disturbing/violent/illegal)
echo "=== MJA Tuning Qwen Eval ==="

# Wait for tuning to finish
while ssh siml-01 "ps aux | grep yhgil99 | grep 'safegen.generate_family' | grep -v grep" 2>/dev/null | grep -q .; do
    sleep 60
done
while ssh siml-02 "ps aux | grep yhgil99 | grep 'safegen.generate_family' | grep -v grep" 2>/dev/null | grep -q .; do
    sleep 60
done

for concept_qwen in "disturbing:shocking" "violent:violence" "illegal:illegal"; do
    concept=${concept_qwen%%:*}; qc=${concept_qwen##*:}
    for d in CAS_SpatialCFG/outputs/v2_experiments/$concept/mja_*/; do
        [ -f "$d/results_qwen3_vl_${qc}.txt" ] && continue
        [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
        echo "[Qwen] $(basename $d)"
        CUDA_VISIBLE_DEVICES=0 python3 vlm/opensource_vlm_i2p_all.py "$d" "$qc" qwen 2>&1 | tail -2
    done
done

echo "=== ALL REMAINING EVAL COMPLETE $(date) ==="
