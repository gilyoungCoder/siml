#!/bin/bash
# v2pack SD1.4 nudity resume — auto-detect start_idx from existing images.
# Usage: bash v2pack_resume_worker.sh <gpu_id> <slot_idx> <n_slots>
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_nudity
mkdir -p $LOGDIR
PACK="$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt"

declare -A PROMPT_FILE EXPECTED
PROMPT_FILE[mma]="$REPO/CAS_SpatialCFG/prompts/mma.txt"
PROMPT_FILE[unlearndiff]="$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt"
EXPECTED[mma]=999
EXPECTED[unlearndiff]=141

# Jobs: DATASET|SS|IMG_THR
JOBS=(
  "mma|10|0.4"
  "mma|15|0.4"
  "mma|15|0.5"
  "mma|20|0.4"
  "unlearndiff|15|0.4"
  "unlearndiff|15|0.5"
  "unlearndiff|20|0.3"
)

N=${#JOBS[@]}
cd $REPO/SafeGen

for ((i=SLOT; i<N; i+=NSLOTS)); do
  IFS='|' read -r DS SS IMG_THR <<< "${JOBS[$i]}"
  prompts="${PROMPT_FILE[$DS]}"
  expected="${EXPECTED[$DS]}"
  outdir="$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity/ours_sd14_v2pack/${DS}/hybrid_ss${SS}_thr0.1_imgthr${IMG_THR}_both"
  LOG="$LOGDIR/resume_g${GPU}_s${SLOT}.log"

  n_imgs=$(ls -1 "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$n_imgs" -ge "$expected" ]; then
    echo "[$(date)] [g$GPU] [skip] ${DS}/ss${SS}/img${IMG_THR} ($n_imgs/$expected)" | tee -a "$LOG"
    continue
  fi

  START_IDX=$n_imgs
  mkdir -p "$outdir"
  echo "[$(date)] [g$GPU] [resume] ${DS}/ss${SS}/img${IMG_THR} from idx=$START_IDX (have $n_imgs/$expected)" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m safegen.generate_family \
    --prompts "$prompts" --outdir "$outdir" \
    --family_guidance --family_config "$PACK" \
    --probe_mode both --how_mode hybrid \
    --cas_threshold 0.6 --safety_scale "$SS" \
    --attn_threshold 0.1 --img_attn_threshold "$IMG_THR" \
    --target_concepts nudity nude_person naked_body \
    --anchor_concepts clothed_person fully_dressed_person \
    --start_idx "$START_IDX" \
    >> "$LOG" 2>&1 || echo "[g$GPU] FAILED ${DS}/ss${SS}/img${IMG_THR}" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU s$SLOT] done"
