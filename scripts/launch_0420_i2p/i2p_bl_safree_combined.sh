#!/bin/bash
# I2P baseline + SAFREE combined worker. Runs both top60 and full_hard variants.
# Usage: bash i2p_bl_safree_combined.sh <gpu> <slot> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
N_SLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

declare -A SF_CONCEPT
SF_CONCEPT[sexual]=nudity
SF_CONCEPT[violence]=violence
SF_CONCEPT[self-harm]=self-harm
SF_CONCEPT[shocking]=shocking
SF_CONCEPT[illegal_activity]=illegal_activity
SF_CONCEPT[harassment]=harassment
SF_CONCEPT[hate]=hate

declare -A FULLHARD_N
FULLHARD_N[sexual]=305
FULLHARD_N[violence]=313
FULLHARD_N[self-harm]=316
FULLHARD_N[shocking]=477
FULLHARD_N[illegal_activity]=238
FULLHARD_N[harassment]=270
FULLHARD_N[hate]=98

CATS=(sexual violence self-harm shocking illegal_activity harassment hate)

# Build jobs: 4 combinations (top60 × fullhard) × (baseline × safree) per category
# Entry format: variant|phase|cat
JOBS=()
for cat in "${CATS[@]}"; do
  JOBS+=("top60|baseline|$cat")
  JOBS+=("top60|safree|$cat")
  JOBS+=("fullhard|baseline|$cat")
  JOBS+=("fullhard|safree|$cat")
done

N=${#JOBS[@]}
echo "[GPU $GPU/slot $SLOT/$N_SLOTS] $N total jobs"

wait_gpu_free() {
  local thr_mb=${1:-40000}
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$thr_mb" ]; then return 0; fi
    sleep 30
  done
}

for ((i=SLOT; i<N; i+=N_SLOTS)); do
  job=${JOBS[$i]}
  IFS='|' read -r VARIANT PHASE CAT <<< "$job"

  if [ "$VARIANT" = "top60" ]; then
    PROMPTS="$REPO/CAS_SpatialCFG/prompts/i2p_sweep60/${CAT}_sweep.txt"
    EXPECTED=60
    OUT_ROOT="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p"
  else
    PROMPTS="$REPO/CAS_SpatialCFG/prompts/i2p_hard/${CAT}_hard.txt"
    EXPECTED=${FULLHARD_N[$CAT]}
    OUT_ROOT="$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_fullhard"
  fi

  if [ "$PHASE" = "baseline" ]; then
    OUTDIR="$OUT_ROOT/baseline_sd14/$CAT"
  else
    OUTDIR="$OUT_ROOT/safree_sd14/$CAT"
  fi

  N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
  if [ "$N_IMGS" -ge "$EXPECTED" ]; then
    echo "[GPU $GPU][skip] $VARIANT $PHASE $CAT ($N_IMGS/$EXPECTED)"
    continue
  fi
  mkdir -p "$OUTDIR"
  echo "[GPU $GPU][run] $VARIANT $PHASE $CAT (expect $EXPECTED)"
  wait_gpu_free 40000

  if [ "$PHASE" = "baseline" ]; then
    cd $REPO
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON CAS_SpatialCFG/generate_baseline.py --nsamples 1 \
      --prompts "$PROMPTS" --outdir "$OUTDIR" --steps 50 \
      >> "$LOGDIR/combined_${VARIANT}_baseline_${CAT}_g${GPU}.log" 2>&1
  else
    SF_CAT=${SF_CONCEPT[$CAT]:-nudity}
    cd $REPO/SAFREE
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_safree_single.py \
      --txt "$PROMPTS" --save-dir "$OUTDIR" \
      --model_id CompVis/stable-diffusion-v1-4 --category $SF_CAT \
      --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
      --seed 42 --image_length 512 --device cuda:0 --erase-id std \
      --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
      --safree -svf -lra --linear_per_prompt_seed \
      >> "$LOGDIR/combined_${VARIANT}_safree_${CAT}_g${GPU}.log" 2>&1
    if [ -d "$OUTDIR/generated" ]; then
      mv "$OUTDIR/generated"/*.png "$OUTDIR"/ 2>/dev/null || true
      rmdir "$OUTDIR/generated" 2>/dev/null || true
    fi
  fi
done
echo "[GPU $GPU/slot $SLOT] combined bl+safree done at $(date)"
