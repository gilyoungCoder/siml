#!/bin/bash
# I2P SD1.4 master worker. Runs all 3 phases on hard subset (sweep60).
# Usage: bash i2p_worker.sh <gpu> <slot> <n_slots>
# Set PHASES env to filter (default: baseline,safree,ours)
set -uo pipefail
GPU=$1
SLOT=$2
N_SLOTS=$3
PHASES=${PHASES:-baseline,safree,ours}
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_sweep60
PACK_DIR=$REPO/CAS_SpatialCFG/exemplars/i2p_v1
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p

declare -A SAFREE_CAT
SAFREE_CAT[sexual]=sexual
SAFREE_CAT[violence]=violence
SAFREE_CAT[self-harm]=self_harm
SAFREE_CAT[shocking]=shocking
SAFREE_CAT[illegal_activity]=illegal
SAFREE_CAT[harassment]=harassment
SAFREE_CAT[hate]=hate

declare -A KW_TGT KW_ANC
KW_TGT[sexual]="nudity nude_person naked_body sexual_content"
KW_ANC[sexual]="clothed_person fully_dressed modest_attire"
KW_TGT[violence]="violence bloody_scene weapon fight"
KW_ANC[violence]="peaceful_scene calm_interaction friendly_gathering"
KW_TGT[self-harm]="self_harm suicide starvation depression"
KW_ANC[self-harm]="healthy_scene wellness happy_life"
KW_TGT[shocking]="shocking horror scream grotesque"
KW_ANC[shocking]="calm_scene peaceful_portrait serene"
KW_TGT[illegal_activity]="drugs crime illegal contraband"
KW_ANC[illegal_activity]="legal_activity daily_life wellness"
KW_TGT[harassment]="harassment mockery intimidation"
KW_ANC[harassment]="respectful friendly supportive"
KW_TGT[hate]="hate discrimination prejudice"
KW_ANC[hate]="unity kindness community"

CATS=(sexual violence self-harm shocking illegal_activity harassment hate)

# Build job list
JOBS=()

# Phase 1: baseline (7 jobs)
if [[ ",$PHASES," == *",baseline,"* ]]; then
  for cat in "${CATS[@]}"; do
    JOBS+=("baseline|$cat|none")
  done
fi
# Phase 2: safree (7 jobs)
if [[ ",$PHASES," == *",safree,"* ]]; then
  for cat in "${CATS[@]}"; do
    JOBS+=("safree|$cat|none")
  done
fi
# Phase 3: ours sweep (252 jobs = 7 cats * 36 cfg)
if [[ ",$PHASES," == *",ours,"* ]]; then
  SS_LIST=(1.5 2.0 2.5)
  THR_LIST=(0.1 0.2)
  HOW_LIST=(anchor hybrid)
  PROBE_LIST=(both imgonly txtonly)
  for cat in "${CATS[@]}"; do
    for ss in "${SS_LIST[@]}"; do
      for thr in "${THR_LIST[@]}"; do
        for how in "${HOW_LIST[@]}"; do
          for pb in "${PROBE_LIST[@]}"; do
            cfg="cas0.6_ss${ss}_thr${thr}_${how}_${pb}"
            JOBS+=("ours|$cat|$ss|$thr|$how|$pb|$cfg")
          done
        done
      done
    done
  done
fi

N=${#JOBS[@]}
echo "[GPU $GPU/slot $SLOT/$N_SLOTS] $N jobs queued"

wait_gpu_free() {
  local thr_mb=${1:-9000}
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$thr_mb" ]; then return 0; fi
    sleep 30
  done
}

for ((i=SLOT; i<N; i+=N_SLOTS)); do
  job=${JOBS[$i]}
  IFS='|' read -ra parts <<< "$job"
  PHASE=${parts[0]}
  CAT=${parts[1]}
  PROMPTS="$PROMPT_DIR/${CAT}_sweep.txt"

  case $PHASE in
    baseline)
      OUTDIR="$OUT_BASE/baseline_sd14/$CAT"
      EXPECTED=$(wc -l < "$PROMPTS")
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] baseline $CAT ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      echo "[GPU $GPU][run] baseline $CAT"
      wait_gpu_free 8000
      cd $REPO
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON CAS_SpatialCFG/generate_baseline.py --nsamples 1 \
        --prompts "$PROMPTS" --outdir "$OUTDIR" --steps 50 \
        >> "$LOGDIR/baseline_${CAT}_g${GPU}.log" 2>&1
      ;;
    safree)
      OUTDIR="$OUT_BASE/safree_sd14/$CAT"
      EXPECTED=$(wc -l < "$PROMPTS")
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] safree $CAT ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      echo "[GPU $GPU][run] safree $CAT"
      wait_gpu_free 8000
      cd $REPO
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON SAFREE/gen_safree_simple.py \
        --safree --svf --lra \
        --txt "$PROMPTS" --outdir "$OUTDIR" \
        --num_images 1 --steps 50 --seed 42 --linear_per_prompt_seed --height 512 --width 512 \
        >> "$LOGDIR/safree_${CAT}_g${GPU}.log" 2>&1
      ;;
    ours)
      SS=${parts[2]}; THR=${parts[3]}; HOW=${parts[4]}; PROBE=${parts[5]}; CFG=${parts[6]}
      OUTDIR="$OUT_BASE/ours_sd14/$CAT/$CFG"
      EXPECTED=$(wc -l < "$PROMPTS")
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] ours $CAT $CFG ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      PACK="$PACK_DIR/$CAT/clip_grouped.pt"
      if [ ! -f "$PACK" ]; then
        echo "[GPU $GPU][miss-pack] $CAT no pack at $PACK, skip"
        continue
      fi
      echo "[GPU $GPU][run] ours $CAT $CFG"
      wait_gpu_free 8000
      cd $REPO/SafeGen
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --probe_mode $PROBE --cas_threshold 0.6 \
        --safety_scale $SS --attn_threshold $THR --img_attn_threshold $THR \
        --how_mode $HOW --family_guidance --family_config "$PACK" \
        >> "$LOGDIR/ours_${CAT}_${CFG}_g${GPU}.log" 2>&1
      ;;
  esac
done
echo "[GPU $GPU/slot $SLOT] done at $(date)"
