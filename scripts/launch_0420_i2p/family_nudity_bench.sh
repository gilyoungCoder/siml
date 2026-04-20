#!/bin/bash
# Family ours on 4 nudity benchmarks with best config (hybrid ss=15, img_thr=0.5, both)
# Usage: bash family_nudity_bench.sh <gpu>
set -uo pipefail
GPU=$1
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

PACK=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity/ours_sd14

# name | prompts | expected
JOBS=(
  "rab|$REPO/CAS_SpatialCFG/prompts/ringabell.txt|79"
  "unlearndiff|$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt|141"
  "p4dn|$REPO/CAS_SpatialCFG/prompts/p4dn.txt|150"
  "mma|$REPO/CAS_SpatialCFG/prompts/mma.txt|999"
)

cd $REPO/SafeGen
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME PROMPTS EXPECTED <<< "$job"
  CFG="hybrid_ss15_thr0.1_imgthr0.5_both"
  OUTDIR="$OUT_BASE/$NAME/$CFG"
  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU][skip] $NAME ($N_IMGS/$EXPECTED)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[GPU $GPU][run] $NAME expect=$EXPECTED"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" --outdir "$OUTDIR" \
    --probe_mode both --cas_threshold 0.6 \
    --safety_scale 15 --attn_threshold 0.1 --img_attn_threshold 0.5 \
    --how_mode hybrid --family_guidance --family_config "$PACK" \
    >> "$LOGDIR/family_nudity_${NAME}_g${GPU}.log" 2>&1
done
echo "[GPU $GPU] family nudity bench done at $(date)"
