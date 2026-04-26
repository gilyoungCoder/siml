#!/bin/bash
# Paper-best dispatcher: reads cells_paper_best.tsv with EXACT paper args.json values per cell.
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
# TSV columns: cell_name, mode, pack_relpath, safety_scale, cas_threshold, attn_threshold,
#              img_attn_threshold, probe_mode, target_concepts(|-sep), anchor_concepts(|-sep),
#              prompts, paper_sr, n_paper, source_dir
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase_tune
LOGDIR=$BASE/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/tune_g${GPU}_s${SLOT}.log
echo "[$(date)] [paper_best g$GPU s$SLOT] worker started" | tee -a $LOG
awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name mode pack ss tau thr_t thr_i probe tgt_str anc_str prompts paper_sr n_paper source_dir; do
  outdir=$OUTBASE/$name
  prompts_abs=$REPO/CAS_SpatialCFG/$prompts
  pack_abs=$REPO/CAS_SpatialCFG/$pack
  if [ ! -f "$prompts_abs" ]; then
    echo "[$(date)] [paper_best g$GPU s$SLOT] MISSING prompts: $prompts_abs" | tee -a $LOG
    continue
  fi
  if [ ! -f "$pack_abs" ]; then
    echo "[$(date)] [paper_best g$GPU s$SLOT] MISSING pack: $pack_abs" | tee -a $LOG
    continue
  fi
  prompt_count=$(wc -l < "$prompts_abs")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [paper_best g$GPU s$SLOT] [skip $name] $existing/$prompt_count" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  # Convert |-separated target/anchor concepts to space-separated CLI args (each may have spaces, so quote)
  IFS='|' read -ra tgt_arr <<< "$tgt_str"
  IFS='|' read -ra anc_arr <<< "$anc_str"
  # Build CLI flags
  tgt_args=""
  for t in "${tgt_arr[@]}"; do
    tgt_args="$tgt_args --__TGT__ \"$t\""
  done
  # Use a python helper to handle quoting properly via array
  echo "[$(date)] [paper_best g$GPU s$SLOT] [run $name] mode=$mode pack=$pack ss=$ss τ=$tau θ=($thr_t,$thr_i) probe=$probe target=$tgt_str start_idx=$existing n_prompts=$prompt_count paper_sr=$paper_sr" | tee -a $LOG
  cd $REPO/SafeGen
  # Use bash array for nargs-style passing
  CMD=( CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family
        --prompts "$prompts_abs" --outdir "$outdir" --start_idx "$existing"
        --family_guidance --family_config "$pack_abs"
        --probe_mode "$probe" --how_mode "$mode"
        --cas_threshold "$tau" --safety_scale "$ss"
        --attn_threshold "$thr_t" --img_attn_threshold "$thr_i"
        --target_concepts "${tgt_arr[@]}"
        --anchor_concepts "${anc_arr[@]}" )
  env "${CMD[@]}" >> $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [paper_best g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [paper_best g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [paper_best g$GPU s$SLOT] worker done" | tee -a $LOG
