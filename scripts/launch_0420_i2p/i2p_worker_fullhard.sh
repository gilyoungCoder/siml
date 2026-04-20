#!/bin/bash
# I2P SD1.4 worker using FULL hard==1 subset per category (not top-60 sweep).
# Usage: bash i2p_worker_fullhard.sh <gpu> <slot> <n_slots>
# Set PHASES env (default: baseline,safree,ours)
set -uo pipefail
GPU=$1
SLOT=$2
N_SLOTS=$3
PHASES=${PHASES:-baseline,safree,ours}
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

PROMPT_DIR=$REPO/CAS_SpatialCFG/prompts/i2p_hard
PACK_DIR=$REPO/CAS_SpatialCFG/exemplars/i2p_v1
OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_i2p_fullhard

declare -A SF_CONCEPT
SF_CONCEPT[sexual]=nudity
SF_CONCEPT[violence]=violence
SF_CONCEPT[self-harm]=self-harm
SF_CONCEPT[shocking]=shocking
SF_CONCEPT[illegal_activity]=illegal_activity
SF_CONCEPT[harassment]=harassment
SF_CONCEPT[hate]=hate

declare -A EXPECTED_MAP
EXPECTED_MAP[sexual]=305
EXPECTED_MAP[violence]=313
EXPECTED_MAP[self-harm]=316
EXPECTED_MAP[shocking]=477
EXPECTED_MAP[illegal_activity]=238
EXPECTED_MAP[harassment]=270
EXPECTED_MAP[hate]=98

CATS=(sexual violence self-harm shocking illegal_activity harassment hate)

JOBS=()
if [[ ",$PHASES," == *",baseline,"* ]]; then
  for cat in "${CATS[@]}"; do JOBS+=("baseline|$cat|none"); done
fi
if [[ ",$PHASES," == *",safree,"* ]]; then
  for cat in "${CATS[@]}"; do JOBS+=("safree|$cat|none"); done
fi
if [[ ",$PHASES," == *",ours,"* ]]; then
  SS_AINP=(1.0 1.5)
  SS_HYB=(10 15 20)
  TXT_THR=0.1
  IMG_THR=0.4
  PROBE_LIST=(both imgonly txtonly)
  for cat in "${CATS[@]}"; do
    for ss in "${SS_AINP[@]}"; do
      for pb in "${PROBE_LIST[@]}"; do
        cfg="cas0.6_ss${ss}_thr${TXT_THR}_imgthr${IMG_THR}_anchor_${pb}"
        JOBS+=("ours|$cat|$ss|$TXT_THR|anchor|$pb|$cfg")
      done
    done
    for ss in "${SS_HYB[@]}"; do
      for pb in "${PROBE_LIST[@]}"; do
        cfg="cas0.6_ss${ss}_thr${TXT_THR}_imgthr${IMG_THR}_hybrid_${pb}"
        JOBS+=("ours|$cat|$ss|$TXT_THR|hybrid|$pb|$cfg")
      done
    done
  done
fi

N=${#JOBS[@]}
echo "[GPU $GPU/slot $SLOT/$N_SLOTS] $N jobs queued (fullhard)"

wait_gpu_free() {
  local thr_mb=${1:-35000}
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
  PROMPTS="$PROMPT_DIR/${CAT}_hard.txt"
  EXPECTED=${EXPECTED_MAP[$CAT]}

  case $PHASE in
    baseline)
      OUTDIR="$OUT_BASE/baseline_sd14/$CAT"
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] baseline $CAT ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      echo "[GPU $GPU][run] baseline $CAT (expect $EXPECTED)"
      wait_gpu_free 40000
      cd $REPO
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON CAS_SpatialCFG/generate_baseline.py --nsamples 1 \
        --prompts "$PROMPTS" --outdir "$OUTDIR" --steps 50 \
        >> "$LOGDIR/fullhard_baseline_${CAT}_g${GPU}.log" 2>&1
      ;;
    safree)
      OUTDIR="$OUT_BASE/safree_sd14/$CAT"
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] safree $CAT ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      SF_CAT=${SF_CONCEPT[$CAT]:-nudity}
      echo "[GPU $GPU][run] safree $CAT cat=$SF_CAT (expect $EXPECTED)"
      wait_gpu_free 40000
      cd $REPO/SAFREE
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_safree_single.py \
        --txt "$PROMPTS" --save-dir "$OUTDIR" \
        --model_id CompVis/stable-diffusion-v1-4 --category $SF_CAT \
        --num-samples 1 --num_inference_steps 50 --guidance_scale 7.5 \
        --seed 42 --image_length 512 --device cuda:0 --erase-id std \
        --sf_alpha 0.01 --re_attn_t=-1,1001 --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2" \
        --safree -svf -lra --linear_per_prompt_seed \
        >> "$LOGDIR/fullhard_safree_${CAT}_g${GPU}.log" 2>&1
      if [ -d "$OUTDIR/generated" ]; then
        mv "$OUTDIR/generated"/*.png "$OUTDIR"/ 2>/dev/null || true
        rmdir "$OUTDIR/generated" 2>/dev/null || true
      fi
      ;;
    ours)
      SS=${parts[2]}; THR=${parts[3]}; HOW=${parts[4]}; PROBE=${parts[5]}; CFG=${parts[6]}
      OUTDIR="$OUT_BASE/ours_sd14/$CAT/$CFG"
      N_IMGS=$(ls -1 "$OUTDIR"/*.png 2>/dev/null | wc -l)
      if [ "$N_IMGS" -ge "$EXPECTED" ]; then
        echo "[GPU $GPU][skip] ours $CAT $CFG ($N_IMGS/$EXPECTED)"
        continue
      fi
      mkdir -p "$OUTDIR"
      PACK="$PACK_DIR/$CAT/clip_grouped.pt"
      if [ ! -f "$PACK" ]; then
        echo "[GPU $GPU][miss-pack] $CAT no pack, skip"
        continue
      fi
      case $PROBE in
        imgonly) PROBE_ARG=image ;;
        txtonly) PROBE_ARG=text ;;
        *) PROBE_ARG=both ;;
      esac
      case $HOW in
        anchor) HOW_ARG=anchor_inpaint ;;
        *) HOW_ARG=hybrid ;;
      esac
      echo "[GPU $GPU][run] ours $CAT $CFG (expect $EXPECTED)"
      wait_gpu_free 40000
      cd $REPO/SafeGen
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --probe_mode $PROBE_ARG --cas_threshold 0.6 \
        --safety_scale $SS --attn_threshold $THR --img_attn_threshold 0.4 \
        --how_mode $HOW_ARG --family_guidance --family_config "$PACK" \
        >> "$LOGDIR/fullhard_ours_${CAT}_${CFG}_g${GPU}.log" 2>&1
      ;;
  esac
done
echo "[GPU $GPU/slot $SLOT] fullhard done at $(date)"
