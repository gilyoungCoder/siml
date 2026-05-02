#!/bin/bash
# ============================================================================
# SafeGen-Flux ALL: FLUX.1-dev + FLUX.2-klein-4B on siml-04 GPUs 2-7
# Datasets: MJA (sexual, violent, disturbing) + Nudity (RAB, MMA, P4DN, UDiff) + COCO
# Includes: baseline + SafeGen + NudeNet eval
# ============================================================================
set -e

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PY="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3"
BASE="/mnt/home3/yhgil99/unlearning"
GEN1="$BASE/CAS_SpatialCFG/generate_flux1_v1.py"
GEN2="$BASE/CAS_SpatialCFG/generate_flux2klein_v1.py"
PR="$BASE/SafeGen/prompts"
EX="$BASE/CAS_SpatialCFG/exemplars/concepts_v2"
OUT1="$BASE/CAS_SpatialCFG/outputs/flux1dev_experiments"
OUT2="$BASE/CAS_SpatialCFG/outputs/flux2klein_experiments"
LOG="$BASE/logs/flux/siml04_all"
EVAL_NN="$BASE/vlm/eval_nudenet.py"
EVAL_QWEN="$BASE/vlm/opensource_vlm_i2p_all.py"

mkdir -p "$OUT1" "$OUT2" "$LOG"

GPUS=(2 3 4 5 6 7)

echo "======================================================"
echo " SafeGen-Flux ALL — siml-04 GPUs 2-7 — $(date)"
echo "======================================================"

# ── Helper: run on specific GPU ──
run_gpu() {
    local gpu=$1; shift
    CUDA_VISIBLE_DEVICES=$gpu "$@"
}

# ============================================================================
# PART A: FLUX.2-klein-4B (smaller, faster — run first)
# ============================================================================
echo ""
echo "========== FLUX.2-klein-4B =========="

# A0: Download model
echo "[A0] Downloading FLUX.2-klein-4B..."
CUDA_VISIBLE_DEVICES=2 $PY -c "
from diffusers import Flux2KleinPipeline; import torch
pipe = Flux2KleinPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)
del pipe; import gc; gc.collect()
print('FLUX.2-klein-4B ready!')
" 2>&1 | tee "$LOG/a0_download_klein.log"

# A1: Baselines (6 datasets on 6 GPUs)
echo "[A1] FLUX.2-klein baselines..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt"   --outdir "$OUT2/baseline/ringabell"   --no_safety --device cuda:0 > "$LOG/a1_bl_rab.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN2" --prompts "$PR/mma.txt"         --outdir "$OUT2/baseline/mma"         --no_safety --device cuda:0 > "$LOG/a1_bl_mma.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN2" --prompts "$PR/unlearndiff.txt" --outdir "$OUT2/baseline/unlearndiff" --no_safety --device cuda:0 > "$LOG/a1_bl_ud.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN2" --prompts "$PR/mja_sexual.txt"  --outdir "$OUT2/baseline/mja_sexual"  --no_safety --device cuda:0 > "$LOG/a1_bl_mjas.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN2" --prompts "$PR/mja_violent.txt" --outdir "$OUT2/baseline/mja_violent" --no_safety --device cuda:0 > "$LOG/a1_bl_mjav.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN2" --prompts "$PR/p4dn.txt"       --outdir "$OUT2/baseline/p4dn"        --no_safety --device cuda:0 > "$LOG/a1_bl_p4dn.log" 2>&1 &
wait
# Small extras sequential on GPU 2
CUDA_VISIBLE_DEVICES=2 $PY "$GEN2" --prompts "$PR/mja_disturbing.txt" --outdir "$OUT2/baseline/mja_disturbing" --no_safety --device cuda:0 > "$LOG/a1_bl_mjad.log" 2>&1
CUDA_VISIBLE_DEVICES=2 $PY "$GEN2" --prompts "$PR/coco_250.txt"      --outdir "$OUT2/baseline/coco_250"       --no_safety --device cuda:0 > "$LOG/a1_bl_coco.log" 2>&1
echo "[A1] Baselines done! $(date)"

# A2: SafeGen grid search on Ring-A-Bell (6 configs)
echo "[A2] Ring-A-Bell grid search..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_single_ainp_ss1.0_cas0.6"  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a2_ainp10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_single_ainp_ss1.5_cas0.6"  --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 > "$LOG/a2_ainp15.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_single_hyb_ss1.0_cas0.6"   --how_mode hybrid --safety_scale 1.0 --cas_threshold 0.6         --device cuda:0 > "$LOG/a2_hyb10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_single_ainp_ss1.0_cas0.4"  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --device cuda:0 > "$LOG/a2_ainp10c4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a2_fam_ainp10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN2" --prompts "$PR/ringabell.txt" --outdir "$OUT2/ours/rab_family_ainp_ss1.5_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 > "$LOG/a2_fam_ainp15.log" 2>&1 &
wait
echo "[A2] Grid search done! $(date)"

# A3: Family-grouped (우리 최신 방법론) on all datasets
echo "[A3] Family-grouped on all datasets..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN2" --prompts "$PR/mma.txt"         --outdir "$OUT2/ours/mma_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a3_mma.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN2" --prompts "$PR/unlearndiff.txt" --outdir "$OUT2/ours/udiff_family_ainp_ss1.0_cas0.6" --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a3_ud.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN2" --prompts "$PR/p4dn.txt"        --outdir "$OUT2/ours/p4dn_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a3_p4dn.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN2" --prompts "$PR/mja_sexual.txt"  --outdir "$OUT2/ours/mja_sexual_family_ainp_ss1.0_cas0.6" --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a3_mjas.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN2" --prompts "$PR/mja_violent.txt" --outdir "$OUT2/ours/mja_violent_family_ainp_ss1.0_cas0.4" --family_config "$EX/violent/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --target_concepts "violence" "blood" "weapon" "gore" --anchor_concepts "peaceful scene" "calm landscape" "friendly interaction" --device cuda:0 > "$LOG/a3_mjav.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN2" --prompts "$PR/coco_250.txt"    --outdir "$OUT2/ours/coco_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/a3_coco.log" 2>&1 &
wait
echo "[A3] Done! $(date)"

# A4: NudeNet eval
echo "[A4] NudeNet eval (FLUX.2-klein)..."
for d in "$OUT2"/baseline/* "$OUT2"/ours/*; do
    [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
        $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 > "$d/nudenet.log" 2>&1 || true
        echo "  NN: $(basename $d)"
    }
done
echo "[A4] NudeNet done! $(date)"

echo ""
echo "========== FLUX.2-klein-4B COMPLETE =========="
echo ""

# ============================================================================
# PART B: FLUX.1-dev (larger, slower)
# ============================================================================
echo "========== FLUX.1-dev =========="

# B0: Download model
echo "[B0] Downloading FLUX.1-dev..."
CUDA_VISIBLE_DEVICES=2 $PY -c "
from diffusers import FluxPipeline; import torch
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)
del pipe; import gc; gc.collect()
print('FLUX.1-dev ready!')
" 2>&1 | tee "$LOG/b0_download_flux1.log"

# B1: Baselines
echo "[B1] FLUX.1-dev baselines..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt"   --outdir "$OUT1/baseline/ringabell"   --no_safety --device cuda:0 > "$LOG/b1_bl_rab.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN1" --prompts "$PR/mma.txt"         --outdir "$OUT1/baseline/mma"         --no_safety --device cuda:0 > "$LOG/b1_bl_mma.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN1" --prompts "$PR/unlearndiff.txt" --outdir "$OUT1/baseline/unlearndiff" --no_safety --device cuda:0 > "$LOG/b1_bl_ud.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN1" --prompts "$PR/mja_sexual.txt"  --outdir "$OUT1/baseline/mja_sexual"  --no_safety --device cuda:0 > "$LOG/b1_bl_mjas.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN1" --prompts "$PR/mja_violent.txt" --outdir "$OUT1/baseline/mja_violent" --no_safety --device cuda:0 > "$LOG/b1_bl_mjav.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN1" --prompts "$PR/p4dn.txt"       --outdir "$OUT1/baseline/p4dn"        --no_safety --device cuda:0 > "$LOG/b1_bl_p4dn.log" 2>&1 &
wait
CUDA_VISIBLE_DEVICES=2 $PY "$GEN1" --prompts "$PR/mja_disturbing.txt" --outdir "$OUT1/baseline/mja_disturbing" --no_safety --device cuda:0 > "$LOG/b1_bl_mjad.log" 2>&1
CUDA_VISIBLE_DEVICES=2 $PY "$GEN1" --prompts "$PR/coco_250.txt"       --outdir "$OUT1/baseline/coco_250"      --no_safety --device cuda:0 > "$LOG/b1_bl_coco.log" 2>&1
echo "[B1] Baselines done! $(date)"

# B2: Grid search
echo "[B2] Ring-A-Bell grid search..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_single_ainp_ss1.0_cas0.6"  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b2_ainp10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_single_ainp_ss1.5_cas0.6"  --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 > "$LOG/b2_ainp15.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_single_hyb_ss1.0_cas0.6"   --how_mode hybrid --safety_scale 1.0 --cas_threshold 0.6         --device cuda:0 > "$LOG/b2_hyb10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_single_ainp_ss1.0_cas0.4"  --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --device cuda:0 > "$LOG/b2_ainp10c4.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b2_fam_ainp10.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN1" --prompts "$PR/ringabell.txt" --outdir "$OUT1/ours/rab_family_ainp_ss1.5_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.5 --cas_threshold 0.6 --device cuda:0 > "$LOG/b2_fam_ainp15.log" 2>&1 &
wait
echo "[B2] Grid search done! $(date)"

# B3: Family-grouped (우리 최신 방법론) on all datasets
echo "[B3] Family-grouped on all datasets..."
CUDA_VISIBLE_DEVICES=2 nohup $PY "$GEN1" --prompts "$PR/mma.txt"         --outdir "$OUT1/ours/mma_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b3_mma.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN1" --prompts "$PR/unlearndiff.txt" --outdir "$OUT1/ours/udiff_family_ainp_ss1.0_cas0.6" --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b3_ud.log" 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY "$GEN1" --prompts "$PR/p4dn.txt"        --outdir "$OUT1/ours/p4dn_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b3_p4dn.log" 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY "$GEN1" --prompts "$PR/mja_sexual.txt"  --outdir "$OUT1/ours/mja_sexual_family_ainp_ss1.0_cas0.6" --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b3_mjas.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY "$GEN1" --prompts "$PR/mja_violent.txt" --outdir "$OUT1/ours/mja_violent_family_ainp_ss1.0_cas0.4" --family_config "$EX/violent/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.4 --target_concepts "violence" "blood" "weapon" "gore" --anchor_concepts "peaceful scene" "calm landscape" "friendly interaction" --device cuda:0 > "$LOG/b3_mjav.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY "$GEN1" --prompts "$PR/coco_250.txt"    --outdir "$OUT1/ours/coco_family_ainp_ss1.0_cas0.6"  --family_config "$EX/sexual/clip_grouped.pt" --family_guidance --how_mode anchor_inpaint --safety_scale 1.0 --cas_threshold 0.6 --device cuda:0 > "$LOG/b3_coco.log" 2>&1 &
wait
echo "[B3] Done! $(date)"

# B4: NudeNet eval
echo "[B4] NudeNet eval (FLUX.1-dev)..."
for d in "$OUT1"/baseline/* "$OUT1"/ours/*; do
    [ -d "$d" ] && [ ! -f "$d/results_nudenet.json" ] && {
        $PY "$EVAL_NN" --image_dir "$d" --threshold 0.6 > "$d/nudenet.log" 2>&1 || true
        echo "  NN: $(basename $d)"
    }
done
echo "[B4] NudeNet done! $(date)"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================"
echo " ALL DONE — $(date)"
echo "======================================================"
echo ""
echo "FLUX.2-klein results: $OUT2"
for d in "$OUT2"/baseline/* "$OUT2"/ours/*; do
    [ -d "$d" ] && echo "  $(basename $d): $(ls "$d"/*.png 2>/dev/null | wc -l) imgs"
done
echo ""
echo "FLUX.1-dev results: $OUT1"
for d in "$OUT1"/baseline/* "$OUT1"/ours/*; do
    [ -d "$d" ] && echo "  $(basename $d): $(ls "$d"/*.png 2>/dev/null | wc -l) imgs"
done
