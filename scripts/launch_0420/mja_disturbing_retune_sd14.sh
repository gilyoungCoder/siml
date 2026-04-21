#!/bin/bash
# MJA-disturbing SD1.4 ours re-tune — best-param focus inspired by sexual family RAB best.
# Uses hybrid + both probe + strong ss (10-20) and separate txt/img thresholds.
# Usage: bash mja_disturbing_retune_sd14.sh <gpu_id> <shard_idx> <num_shards>
set -uo pipefail
GPU=$1
SHARD=$2
NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420
mkdir -p $LOGDIR
LOG="$LOGDIR/mja_disturbing_retune_g${GPU}_s${SHARD}.log"

# 5 configs — all hybrid + both probe, inspired by sexual family RAB best (hybrid ss20 imgthr0.4)
JOBS=(
  "10|0.1|0.4"
  "15|0.1|0.4"
  "20|0.1|0.4"
  "15|0.1|0.5"
  "20|0.1|0.3"
)

N=${#JOBS[@]}
PROMPTS="$REPO/CAS_SpatialCFG/prompts/mja_disturbing.txt"
CONFIG_PATH="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/disturbing/clip_grouped.pt"
TGT_KW="grotesque body_horror gore monster"
ANC_KW="beautiful_art peaceful_scene"

cd $REPO/SafeGen

echo "[$(date)] [retune g$GPU s$SHARD/$NSHARDS] total $N configs, shard owns indices $SHARD,$((SHARD+NSHARDS)),..." | tee -a "$LOG"

for ((i=SHARD; i<N; i+=NSHARDS)); do
  IFS='|' read -r SS TXT_THR IMG_THR <<< "${JOBS[$i]}"
  OUTDIR="$REPO/CAS_SpatialCFG/outputs/launch_0420/ours_sd14/mja_disturbing/hybrid_ss${SS}_thr${TXT_THR}_imgthr${IMG_THR}_both_retune"

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge 100 ]; then
    echo "[$(date)] [g$GPU] [skip] cfg#$i ss=$SS thr=$TXT_THR imgthr=$IMG_THR ($N_IMGS imgs)" | tee -a "$LOG"
    continue
  fi

  mkdir -p "$OUTDIR"
  echo "[$(date)] [g$GPU] [run #$i] ss=$SS txt_thr=$TXT_THR img_thr=$IMG_THR -> $OUTDIR" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --family_guidance \
    --family_config "$CONFIG_PATH" \
    --probe_mode both \
    --how_mode hybrid \
    --cas_threshold 0.6 \
    --safety_scale "$SS" \
    --attn_threshold "$TXT_THR" \
    --img_attn_threshold "$IMG_THR" \
    --target_concepts $TGT_KW \
    --anchor_concepts $ANC_KW \
    >> "$LOG" 2>&1 || echo "[g$GPU] cfg#$i FAILED" | tee -a "$LOG"
done

echo "[$(date)] [retune g$GPU s$SHARD] done" | tee -a "$LOG"
