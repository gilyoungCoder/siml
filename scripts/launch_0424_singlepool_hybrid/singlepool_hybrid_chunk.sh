#!/bin/bash
# Single-pool HYBRID generation chunk — one (concept, gpu, start, end) cell.
# Uses --clip_embeddings (NOT --family_config), --no-family-guidance,
# matching family hybrid best hyperparameters (ss=22 tt=0.15 ti=0.1 cas=0.6 both probe).
# Usage: bash singlepool_hybrid_chunk.sh <gpu> <concept> <start> <end>
set -uo pipefail
GPU=$1; CONCEPT=$2; START=$3; END=$4
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/$CONCEPT/clip_exemplar_projected.pt
PROMPTS=$REPO/CAS_SpatialCFG/prompts/mja_${CONCEPT}.txt
OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0424_singlepool_hybrid/mja_${CONCEPT}/hybrid_ss22_tt0.15_ti0.1_both_cas0.6
LOGDIR=$REPO/logs/launch_0424_singlepool_hybrid
LOG=$LOGDIR/g${GPU}_${CONCEPT}_${START}_${END}.log
mkdir -p "$OUTDIR" "$LOGDIR"

# Concept-correct target/anchor concepts
case "$CONCEPT" in
  sexual)     TC="nudity nude_person naked_body";              AC="clothed_person fully_dressed_person" ;;
  violent)    TC="violence bloody_scene weapon fight";         AC="peaceful_scene calm_interaction" ;;
  illegal)    TC="drugs crime contraband illegal_activity";    AC="legal_activity daily_life" ;;
  disturbing) TC="grotesque body_horror gore monster";          AC="serene_scene wholesome_art" ;;
esac

cd $REPO/SafeGen
echo "[$(date)] [g$GPU] $CONCEPT/hybrid/single chunk $START-$END" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
  --prompts "$PROMPTS" --outdir "$OUTDIR" \
  --clip_embeddings "$PACK" \
  --probe_mode both --how_mode hybrid \
  --cas_threshold 0.6 --safety_scale 22 \
  --attn_threshold 0.15 --img_attn_threshold 0.1 \
  --target_concepts $TC --anchor_concepts $AC \
  --start_idx $START --end_idx $END \
  >> "$LOG" 2>&1 || echo "[g$GPU] FAILED" | tee -a "$LOG"
echo "[$(date)] [g$GPU] $CONCEPT chunk $START-$END done" | tee -a "$LOG"
