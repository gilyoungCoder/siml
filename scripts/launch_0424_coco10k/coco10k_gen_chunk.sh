#!/bin/bash
# COCO 10k generation chunk worker — single (mode, gpu, start, end) cell.
# Mode: baseline | anchor | hybrid
# Usage: bash coco10k_gen_chunk.sh <gpu> <mode> <start> <end>
set -uo pipefail
GPU=$1; MODE=$2; START=$3; END=$4
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/coco_10k.txt

LOGDIR=$REPO/logs/launch_0424_coco10k
LOG=$LOGDIR/g${GPU}_${MODE}_${START}_${END}.log
mkdir -p "$LOGDIR"

case "$MODE" in
  baseline)
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0424_coco10k/baseline_sd14
    EXTRA=""
    ;;
  anchor)
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0424_coco10k/ours_anchor_v2pack
    PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt
    EXTRA="--family_guidance --family_config $PACK --probe_mode both --how_mode anchor_inpaint --cas_threshold 0.6 --safety_scale 1.2 --attn_threshold 0.1 --img_attn_threshold 0.3 --target_concepts nudity nude_person naked_body --anchor_concepts clothed_person fully_dressed_person"
    ;;
  hybrid)
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0424_coco10k/ours_hybrid_v1pack
    PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt
    EXTRA="--family_guidance --family_config $PACK --probe_mode both --how_mode hybrid --cas_threshold 0.6 --safety_scale 20 --attn_threshold 0.1 --img_attn_threshold 0.3 --target_concepts nudity nude_person naked_body --anchor_concepts clothed_person fully_dressed_person"
    ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

mkdir -p "$OUTDIR"
cd $REPO/SafeGen
echo "[$(date)] [g$GPU] $MODE chunk $START-$END start" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
  --prompts "$PROMPTS" --outdir "$OUTDIR" \
  --start_idx $START --end_idx $END \
  $EXTRA \
  >> "$LOG" 2>&1 || echo "[g$GPU] FAILED" | tee -a "$LOG"

echo "[$(date)] [g$GPU] $MODE chunk $START-$END done" | tee -a "$LOG"
