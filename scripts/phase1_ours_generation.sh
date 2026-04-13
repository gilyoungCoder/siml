#!/bin/bash
# Phase 1: Generate Ours (SafeGen) images across all datasets
# Runs on siml-01 GPU 0-7 and siml-02 GPU 0-6
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy

SAFEGEN="SafeGen/safegen"
EXEMPLAR_BASE="CAS_SpatialCFG/exemplars/concepts_v2"
OUT_BASE="CAS_SpatialCFG/outputs/v2_experiments"
mkdir -p $OUT_BASE

# Wait for exemplar .pt files to be ready
echo "Checking exemplar .pt files..."
for concept in sexual violent disturbing illegal harassment hate selfharm; do
    pt="${EXEMPLAR_BASE}/${concept}/clip_grouped.pt"
    while [ ! -f "$pt" ]; do
        echo "Waiting for $pt ..."
        sleep 30
    done
    echo "  OK: $pt"
done
echo "All exemplar .pt files ready!"

run_ours() {
    local name=$1
    local prompts=$2
    local gpu=$3
    local concept=$4
    local probe=$5       # text / image / both
    local how=$6         # anchor_inpaint / hybrid / target_sub
    local family=$7      # 1=family, 0=single
    local cas=$8
    local ss=$9
    local target_concepts="${10}"
    local anchor_concepts="${11}"

    local suffix="${probe}_${how}"
    [ "$family" = "1" ] && suffix="${suffix}_fam" || suffix="${suffix}_single"
    suffix="${suffix}_cas${cas}_ss${ss}"
    local outdir="${OUT_BASE}/${concept}/${name}_${suffix}"

    if [ -d "$outdir" ] && [ "$(ls -1 $outdir/*.png 2>/dev/null | wc -l)" -gt 5 ]; then
        echo "[SKIP] $outdir already has images"
        return
    fi

    local family_args=""
    if [ "$family" = "1" ]; then
        family_args="--family_guidance --family_config ${EXEMPLAR_BASE}/${concept}/clip_grouped.pt"
    else
        family_args="--clip_embeddings ${EXEMPLAR_BASE}/${concept}/clip_exemplar_projected.pt"
    fi

    # For text-only mode, don't need clip embeddings
    if [ "$probe" = "text" ]; then
        family_args=""
        [ "$family" = "1" ] && family_args="--family_guidance --family_config ${EXEMPLAR_BASE}/${concept}/clip_grouped.pt"
    fi

    echo "[GPU $gpu] ${concept}/${name}_${suffix}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_family \
        --prompts "$prompts" \
        --outdir "$outdir" \
        --probe_mode "$probe" \
        --how_mode "$how" \
        --cas_threshold "$cas" \
        --safety_scale "$ss" \
        --target_concepts $target_concepts \
        --anchor_concepts $anchor_concepts \
        --steps 50 --seed 42 --cfg_scale 7.5 \
        $family_args \
        2>&1 | tail -5

    echo "  Done: $outdir ($(ls -1 $outdir/*.png 2>/dev/null | wc -l) images)"
}

# ============================================================
# NUDITY DATASETS (4 datasets)
# ============================================================
echo ""
echo "=========================================="
echo "  NUDITY EXPERIMENTS"
echo "=========================================="

NUDE_TGT="nudity nude naked exposed"
NUDE_ANC="clothed dressed covered wearing_clothes"

# Ring-A-Bell (79 prompts) - grid search
for how in anchor_inpaint hybrid; do
    for probe in text image both; do
        for fam in 0 1; do
            for cas in 0.4 0.6; do
                for ss in 1.0 1.2 1.5; do
                    run_ours "rab" "CAS_SpatialCFG/prompts/ringabell.txt" 0 "sexual" "$probe" "$how" "$fam" "$cas" "$ss" "$NUDE_TGT" "$NUDE_ANC" &
                    # Throttle: max 2 jobs per GPU
                    if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
                done
            done
        done
    done
done
wait

# MMA (1000 prompts) - best configs only
for how in anchor_inpaint hybrid; do
    for probe in both text; do
        run_ours "mma" "CAS_SpatialCFG/prompts/mma.txt" 1 "sexual" "$probe" "$how" 1 0.6 1.2 "$NUDE_TGT" "$NUDE_ANC" &
    done
done
wait

# P4DN (151 prompts)
for probe in both text; do
    run_ours "p4dn" "CAS_SpatialCFG/prompts/p4dn.txt" 2 "sexual" "$probe" "anchor_inpaint" 1 0.6 1.2 "$NUDE_TGT" "$NUDE_ANC" &
done
wait

# UnlearnDiff (142 prompts)
for probe in both text; do
    run_ours "udiff" "CAS_SpatialCFG/prompts/unlearndiff.txt" 2 "sexual" "$probe" "anchor_inpaint" 1 0.6 1.2 "$NUDE_TGT" "$NUDE_ANC" &
done
wait

# ============================================================
# MJA DATASETS (4 categories × grid)
# ============================================================
echo ""
echo "=========================================="
echo "  MJA EXPERIMENTS"
echo "=========================================="

# Violence (MJA violent, 100 prompts)
VIOL_TGT="blood knife gun weapon wound fight"
VIOL_ANC="clean toy_gun handshake healthy"

for how in anchor_inpaint hybrid; do
    for probe in text image both; do
        for fam in 0 1; do
            for cas in 0.4 0.5; do
                for ss in 1.0 1.5; do
                    run_ours "mja" "CAS_SpatialCFG/prompts/mja_violent.txt" 3 "violent" "$probe" "$how" "$fam" "$cas" "$ss" "$VIOL_TGT" "$VIOL_ANC" &
                    if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
                done
            done
        done
    done
done
wait

# Disturbing (MJA disturbing, 100 prompts)
DIST_TGT="grotesque horror monster demon scream blood"
DIST_ANC="beautiful gentle friendly calm smile"

for probe in text image both; do
    for fam in 0 1; do
        run_ours "mja" "CAS_SpatialCFG/prompts/mja_disturbing.txt" 4 "disturbing" "$probe" "anchor_inpaint" "$fam" 0.5 1.0 "$DIST_TGT" "$DIST_ANC" &
        if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
    done
done
wait

# Sexual (MJA sexual, 100 prompts)
for probe in text image both; do
    for fam in 0 1; do
        run_ours "mja" "CAS_SpatialCFG/prompts/mja_sexual.txt" 5 "sexual" "$probe" "anchor_inpaint" "$fam" 0.6 1.2 "$NUDE_TGT" "$NUDE_ANC" &
        if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
    done
done
wait

# Illegal (MJA illegal, 100 prompts)
ILLE_TGT="drug pills marijuana smoke trafficking"
ILLE_ANC="medicine vitamins tea coffee portrait"

for probe in text image both; do
    for fam in 0 1; do
        run_ours "mja" "CAS_SpatialCFG/prompts/mja_illegal.txt" 6 "illegal" "$probe" "anchor_inpaint" "$fam" 0.5 1.0 "$ILLE_TGT" "$ILLE_ANC" &
        if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
    done
done
wait

echo ""
echo "=========================================="
echo "  Phase 1 COMPLETE"
echo "=========================================="
