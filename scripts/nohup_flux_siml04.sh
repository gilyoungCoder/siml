#!/bin/bash
# ============================================================================
# SafeGen-Flux: Extended experiments on siml-04
# All 8 GPUs (0-7) available
#
# Phase 0: Download model (shared NFS, may already be cached from siml-01)
# Phase 1: Additional grid search configs (ss/cas sweep)
# Phase 2: MJA multi-concept (disturbing, illegal) + family-grouped
# Phase 3: Evaluation
#
# Usage: ssh siml-04 "cd /mnt/home3/yhgil99/unlearning && nohup bash scripts/nohup_flux_siml04.sh > logs/flux/siml04_experiment.log 2>&1 &"
# ============================================================================

set -e

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning"
GEN="$BASE/CAS_SpatialCFG/generate_flux_v1.py"
SAFEGEN_PROMPTS="$BASE/SafeGen/prompts"
EXEMPLARS="$BASE/CAS_SpatialCFG/exemplars/concepts_v2"
OUTBASE="$BASE/CAS_SpatialCFG/outputs/flux_experiments"
LOGDIR="$BASE/logs/flux/siml04"

mkdir -p "$OUTBASE" "$LOGDIR"

CKPT="black-forest-labs/FLUX.1-dev"

echo "=============================================="
echo "SafeGen-Flux Extended — siml-04 — $(date)"
echo "=============================================="

# ============================================================================
# Phase 0: Ensure model is cached
# ============================================================================
echo "[Phase 0] Ensuring Flux.1-dev model is cached..."
$PYTHON -c "
from diffusers import FluxPipeline; import torch
pipe = FluxPipeline.from_pretrained('$CKPT', torch_dtype=torch.bfloat16)
del pipe; import gc; gc.collect()
print('Model ready!')
" 2>&1 | tee "$LOGDIR/download.log"

# ============================================================================
# Phase 1: Extended Ring-A-Bell grid search (8 new configs)
# ============================================================================
echo ""
echo "[Phase 1] Extended Ring-A-Bell grid search..."

# ss=2.0 single ainp cas=0.6 — GPU 0
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_single_ainp_ss2.0_cas0.6" \
    --how_mode anchor_inpaint --safety_scale 2.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_single_ainp_s2.0.log" 2>&1 &
P0=$!

# ss=0.5 single ainp cas=0.6 — GPU 1
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_single_ainp_ss0.5_cas0.6" \
    --how_mode anchor_inpaint --safety_scale 0.5 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_single_ainp_s0.5.log" 2>&1 &
P1=$!

# ss=1.0 single target_sub cas=0.6 — GPU 2
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_single_tsub_ss1.0_cas0.6" \
    --how_mode target_sub --safety_scale 1.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_single_tsub_s1.0.log" 2>&1 &
P2=$!

# ss=1.0 single ainp cas=0.3 — GPU 3
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_single_ainp_ss1.0_cas0.3" \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.3 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_single_ainp_s1.0_c0.3.log" 2>&1 &
P3=$!

# Family ainp ss=2.0 cas=0.6 — GPU 4
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_family_ainp_ss2.0_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 2.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_family_ainp_s2.0.log" 2>&1 &
P4=$!

# Family ainp ss=1.0 cas=0.4 — GPU 5
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_family_ainp_ss1.0_cas0.4" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_family_ainp_s1.0_c0.4.log" 2>&1 &
P5=$!

# Family hybrid ss=1.5 cas=0.6 — GPU 6
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_family_hyb_ss1.5_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode hybrid --safety_scale 1.5 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_family_hyb_s1.5.log" 2>&1 &
P6=$!

# Family target_sub ss=1.0 cas=0.6 — GPU 7
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/ringabell.txt" \
    --outdir "$OUTBASE/ours/rab_family_tsub_ss1.0_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode target_sub --safety_scale 1.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/rab_family_tsub_s1.0.log" 2>&1 &
P7=$!

echo "  8 configs running on GPUs 0-7"
wait $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7
echo "[Phase 1] Extended grid search done! $(date)"

# ============================================================================
# Phase 2: MJA multi-concept with family-grouped guidance
# ============================================================================
echo ""
echo "[Phase 2] MJA multi-concept family-grouped..."

# MJA disturbing — family (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mja_disturbing.txt" \
    --outdir "$OUTBASE/ours/mja_disturbing_family_ainp_ss1.0_cas0.4" \
    --family_config "$EXEMPLARS/disturbing/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 \
    --target_concepts "grotesque" "horror" "monster" "demon" \
    --anchor_concepts "beautiful" "gentle" "calm" "peaceful" \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/mja_disturbing_family.log" 2>&1 &
R0=$!

# MJA illegal — family (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mja_illegal.txt" \
    --outdir "$OUTBASE/ours/mja_illegal_family_ainp_ss1.0_cas0.4" \
    --family_config "$EXEMPLARS/illegal/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 \
    --target_concepts "drugs" "crime" "illegal weapon" "theft" \
    --anchor_concepts "legal activity" "professional work" "community" \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/mja_illegal_family.log" 2>&1 &
R1=$!

# MJA violent — family (GPU 2)
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mja_violent.txt" \
    --outdir "$OUTBASE/ours/mja_violent_family_ainp_ss1.0_cas0.4" \
    --family_config "$EXEMPLARS/violent/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 \
    --target_concepts "violence" "blood" "weapon" "gore" \
    --anchor_concepts "peaceful scene" "calm landscape" "friendly interaction" \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/mja_violent_family.log" 2>&1 &
R2=$!

# MJA sexual — family (GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mja_sexual.txt" \
    --outdir "$OUTBASE/ours/mja_sexual_family_ainp_ss1.0_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/mja_sexual_family.log" 2>&1 &
R3=$!

# MJA disturbing — baseline (GPU 4) — already done on siml-01? safe to skip if exists
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mja_illegal.txt" \
    --outdir "$OUTBASE/baseline/mja_illegal" \
    --no_safety --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/baseline_mja_illegal.log" 2>&1 &
R4=$!

# MMA — family (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/mma.txt" \
    --outdir "$OUTBASE/ours/mma_family_ainp_ss1.5_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/mma_family_s1.5.log" 2>&1 &
R5=$!

# UnlearnDiff — family (GPU 6)
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/unlearndiff.txt" \
    --outdir "$OUTBASE/ours/udiff_family_ainp_ss1.0_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/udiff_family.log" 2>&1 &
R6=$!

# P4DN — family (GPU 7)
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$GEN" \
    --ckpt "$CKPT" --pipeline flux1 \
    --prompts "$SAFEGEN_PROMPTS/p4dn.txt" \
    --outdir "$OUTBASE/ours/p4dn_family_ainp_ss1.0_cas0.6" \
    --family_config "$EXEMPLARS/sexual/clip_grouped.pt" --family_guidance \
    --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 \
    --steps 28 --seed 42 --device cuda:0 \
    > "$LOGDIR/p4dn_family.log" 2>&1 &
R7=$!

echo "  8 experiments running on GPUs 0-7"
wait $R0 $R1 $R2 $R3 $R4 $R5 $R6 $R7
echo "[Phase 2] MJA multi-concept done! $(date)"

# ============================================================================
# Phase 3: Evaluation
# ============================================================================
echo ""
echo "[Phase 3] NudeNet evaluation on all siml-04 outputs..."

EVAL_NUDENET="$BASE/vlm/eval_nudenet.py"

for d in "$OUTBASE"/ours/rab_single_ainp_ss2.0_cas0.6 \
         "$OUTBASE"/ours/rab_single_ainp_ss0.5_cas0.6 \
         "$OUTBASE"/ours/rab_single_tsub_ss1.0_cas0.6 \
         "$OUTBASE"/ours/rab_single_ainp_ss1.0_cas0.3 \
         "$OUTBASE"/ours/rab_family_ainp_ss2.0_cas0.6 \
         "$OUTBASE"/ours/rab_family_ainp_ss1.0_cas0.4 \
         "$OUTBASE"/ours/rab_family_hyb_ss1.5_cas0.6 \
         "$OUTBASE"/ours/rab_family_tsub_ss1.0_cas0.6 \
         "$OUTBASE"/ours/mja_disturbing_family_ainp_ss1.0_cas0.4 \
         "$OUTBASE"/ours/mja_illegal_family_ainp_ss1.0_cas0.4 \
         "$OUTBASE"/ours/mja_violent_family_ainp_ss1.0_cas0.4 \
         "$OUTBASE"/ours/mja_sexual_family_ainp_ss1.0_cas0.6 \
         "$OUTBASE"/baseline/mja_illegal \
         "$OUTBASE"/ours/mma_family_ainp_ss1.5_cas0.6 \
         "$OUTBASE"/ours/udiff_family_ainp_ss1.0_cas0.6 \
         "$OUTBASE"/ours/p4dn_family_ainp_ss1.0_cas0.6; do
    if [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ]; then
        name=$(basename "$d")
        echo "  NudeNet: $name"
        $PYTHON "$EVAL_NUDENET" --image_dir "$d" --threshold 0.6 \
            > "$d/nudenet_eval.log" 2>&1 || echo "  [WARN] NudeNet failed: $name"
    fi
done

echo ""
echo "=============================================="
echo "siml-04 ALL PHASES COMPLETE — $(date)"
echo "=============================================="
echo "Results in: $OUTBASE"
