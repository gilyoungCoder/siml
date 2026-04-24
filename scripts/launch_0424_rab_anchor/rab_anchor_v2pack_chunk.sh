#!/bin/bash
# RAB anchor v2pack uniform-config — single chunk worker
# Same config as UD/MMA/P4DN: anchor_inpaint, ss=1.2, τ_t=0.1, τ_i=0.3, both probe, concepts_v2/sexual pack.
# Usage: bash rab_anchor_v2pack_chunk.sh <gpu> <start> <end>
set -uo pipefail
GPU=$1; START=$2; END=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=$REPO/../.conda/envs/sdd_copy/bin/python3.10
[ -x "$PY" ] || PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt
outdir=$REPO/CAS_SpatialCFG/outputs/launch_0424_rab_anchor_v2pack/anchor_ss1.2_thr0.1_imgthr0.3_both
LOG=$REPO/logs/launch_0424_rab_anchor/g${GPU}_${START}_${END}.log
mkdir -p "$outdir" "$REPO/logs/launch_0424_rab_anchor"
cd $REPO/SafeGen
echo "[$(date)] [g$GPU chunk $START-$END] start" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
  --prompts $REPO/CAS_SpatialCFG/prompts/ringabell.txt --outdir "$outdir" \
  --family_guidance --family_config "$PACK" \
  --probe_mode both --how_mode anchor_inpaint \
  --cas_threshold 0.6 --safety_scale 1.2 \
  --attn_threshold 0.1 --img_attn_threshold 0.3 \
  --target_concepts nudity nude_person naked_body \
  --anchor_concepts clothed_person fully_dressed_person \
  --start_idx $START --end_idx $END \
  >> "$LOG" 2>&1 || echo "[g$GPU] FAILED" | tee -a "$LOG"
echo "[$(date)] [g$GPU chunk $START-$END] done" | tee -a "$LOG"
