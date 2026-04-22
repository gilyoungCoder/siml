#!/bin/bash
# SD3 hybrid ss>=15 with concept-tuned CAS (violent 0.5 / illegal 0.45 / disturbing 0.6)
# Usage: bash worker.sh <GPU> <SLOT> <NSHARDS>
set -uo pipefail
GPU=$1; SLOT=$2; NSHARDS=$3

REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0423_sd3_hybrid
mkdir -p $LOGDIR
LOG=$LOGDIR/g${GPU}_s${SLOT}.log

# dataset|cas|ss|pack_subdir|eval_concept
JOBS=(
  "mja_violent|0.5|15|concepts_v2/violent|violence"
  "mja_illegal|0.45|15|concepts_v2/illegal|illegal"
  "mja_disturbing|0.6|15|concepts_v2/disturbing|disturbing"
  "mja_violent|0.5|20|concepts_v2/violent|violence"
  "mja_illegal|0.45|20|concepts_v2/illegal|illegal"
  "mja_disturbing|0.6|20|concepts_v2/disturbing|disturbing"
  "mja_violent|0.5|25|concepts_v2/violent|violence"
  "mja_illegal|0.45|25|concepts_v2/illegal|illegal"
  "mja_disturbing|0.6|25|concepts_v2/disturbing|disturbing"
)

N=${#JOBS[@]}
cd $REPO
for ((i=SLOT; i<N; i+=NSHARDS)); do
  IFS='|' read -r DS CAS SS PACK_DIR EVAL_CONCEPT <<< "${JOBS[$i]}"
  PACK=$REPO/CAS_SpatialCFG/exemplars/${PACK_DIR}/clip_grouped.pt
  PROMPTS=$REPO/CAS_SpatialCFG/prompts/${DS}.txt
  CFG="cas${CAS}_ss${SS}_thr0.1_imgthr0.1_hybrid_both"
  OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd3/${DS}/${CFG}
  mkdir -p "$OUTDIR"
  n=$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$n" -ge 100 ]; then
    echo "[skip gen] $DS cas=$CAS ss=$SS ($n imgs)" | tee -a "$LOG"
  else
    echo "[$(date)] [g$GPU s$SLOT] [gen] $DS cas=$CAS ss=$SS" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/sd3/generate_sd3_safegen.py \
      --prompts "$PROMPTS" --outdir "$OUTDIR" \
      --family_guidance --family_config "$PACK" \
      --probe_mode both --how_mode hybrid \
      --cas_threshold "$CAS" --safety_scale "$SS" \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      >> "$LOG" 2>&1 || { echo "FAIL gen $DS cas=$CAS ss=$SS" | tee -a "$LOG"; continue; }
  fi
  EVAL_DONE="$OUTDIR/.eval_v5_qwen3_vl_${EVAL_CONCEPT}.done"
  if [ -f "$EVAL_DONE" ]; then
    echo "[skip eval] $DS $EVAL_CONCEPT" | tee -a "$LOG"
  else
    echo "[$(date)] [g$GPU s$SLOT] [eval] $DS $EVAL_CONCEPT" | tee -a "$LOG"
    cd $REPO/vlm
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON opensource_vlm_i2p_all_v5.py "$OUTDIR" "$EVAL_CONCEPT" qwen >> "$LOG" 2>&1 \
      && touch "$EVAL_DONE"
    cd $REPO
  fi
done
echo "[$(date)] [g$GPU s$SLOT] DONE" | tee -a "$LOG"
