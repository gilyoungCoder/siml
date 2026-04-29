#!/bin/bash
# Hate v2 worker: τ=0.6 fixed, output to phase_hate_v2.
# Args: $1=GPU $2=SLOT $3=NSLOTS $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
PACK=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/hate/clip_grouped.pt
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60/hate_sweep.txt
OUTBASE=$BASE/outputs/phase_hate_v2
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/hate_v2_g${GPU}_s${SLOT}.log
echo "[$(date)] hate v2 worker GPU=$GPU SLOT=$SLOT NSLOTS=$NSLOTS" | tee -a $LOG

awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name desc sh tau attn_t img_attn n_tok target anchor_concepts; do
  outdir=$OUTBASE/$name
  prompt_count=$(wc -l < "$PROMPTS")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name ($existing/$prompt_count)" >> $LOG
    continue
  fi
  mkdir -p "$outdir"
  IFS='|' read -ra tgt_arr <<< "$target"
  IFS='|' read -ra anc_arr <<< "$anchor_concepts"
  echo "[$(date +%H:%M:%S)] LAUNCH $name desc=$desc sh=$sh τ=$tau attn=$attn_t img=$img_attn" >> $LOG
  cd $REPO/SafeGen
  CMD=( CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family
        --prompts "$PROMPTS" --outdir "$outdir" --start_idx "$existing"
        --family_guidance --family_config "$PACK"
        --probe_mode both --how_mode hybrid --probe_fusion union
        --cas_threshold "$tau" --safety_scale "$sh"
        --attn_threshold "$attn_t" --img_attn_threshold "$img_attn"
        --n_img_tokens "$n_tok"
        --steps 50 --seed 42 --cfg_scale 7.5
        --target_concepts "${tgt_arr[@]}"
        --anchor_concepts "${anc_arr[@]}" )
  env "${CMD[@]}" >> $LOG 2>&1
  rc=$?
  final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] FAIL $name exit=$rc imgs=$final" >> $LOG
  else
    echo "[$(date +%H:%M:%S)] DONE $name imgs=$final" >> $LOG
  fi
done
echo "[$(date)] hate v2 worker done GPU=$GPU SLOT=$SLOT" | tee -a $LOG
