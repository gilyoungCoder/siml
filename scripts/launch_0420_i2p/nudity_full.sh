#!/bin/bash
# Full nudity benchmark: baseline + ours (5 configs × 2 packs × 4 datasets)
# Usage: bash nudity_full.sh <gpu> <slot> <n_slots>
set -uo pipefail
GPU=$1; SLOT=$2; N_SLOTS=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0420_i2p
mkdir -p $LOGDIR

declare -A PROMPTS_FILE EXPECTED
PROMPTS_FILE[rab]=$REPO/CAS_SpatialCFG/prompts/ringabell.txt; EXPECTED[rab]=79
PROMPTS_FILE[unlearndiff]=$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt; EXPECTED[unlearndiff]=141
PROMPTS_FILE[p4dn]=$REPO/CAS_SpatialCFG/prompts/p4dn.txt; EXPECTED[p4dn]=150
PROMPTS_FILE[mma]=$REPO/CAS_SpatialCFG/prompts/mma.txt; EXPECTED[mma]=999

declare -A PACK_PATH
PACK_PATH[v1]=$REPO/CAS_SpatialCFG/exemplars/i2p_v1/sexual/clip_grouped.pt
PACK_PATH[v2]=$REPO/CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt

OUT_BASE=$REPO/CAS_SpatialCFG/outputs/launch_0420_nudity

# Build jobs: baseline (4) + ours (5 cfg × 2 pack × 4 ds = 40)
JOBS=()
for ds in rab unlearndiff p4dn mma; do
  JOBS+=("baseline|$ds|none")
done
for pack in v1 v2; do
  for cfg in "hybrid:10:0.3" "hybrid:10:0.4" "hybrid:15:0.4" "hybrid:15:0.5" "hybrid:20:0.3"; do
    IFS=: read -r HOW SS IMG <<< "$cfg"
    for ds in rab unlearndiff p4dn mma; do
      JOBS+=("ours|$ds|$pack|$HOW|$SS|$IMG")
    done
  done
done

N=${#JOBS[@]}
echo "[GPU $GPU/slot $SLOT/$N_SLOTS] $N total jobs"

for ((i=SLOT; i<N; i+=N_SLOTS)); do
  job=${JOBS[$i]}
  IFS='|' read -ra parts <<< "$job"
  PHASE=${parts[0]}
  DS=${parts[1]}
  PROMPTS=${PROMPTS_FILE[$DS]}
  EXP=${EXPECTED[$DS]}

  if [ "$PHASE" = "baseline" ]; then
    OUT=$OUT_BASE/baseline_sd14/$DS
    N_IMGS=$(ls -1 "$OUT"/*.png 2>/dev/null | wc -l)
    if [ "$N_IMGS" -ge "$EXP" ]; then echo "[GPU $GPU][skip] baseline $DS ($N_IMGS/$EXP)"; continue; fi
    mkdir -p "$OUT"
    echo "[GPU $GPU][run] baseline $DS exp=$EXP"
    cd $REPO
    CUDA_VISIBLE_DEVICES=$GPU $PY CAS_SpatialCFG/generate_baseline.py --nsamples 1 \
      --prompts "$PROMPTS" --outdir "$OUT" --steps 50 \
      >> "$LOGDIR/nudity_baseline_${DS}_g${GPU}.log" 2>&1
  else
    PACK_KEY=${parts[2]}; HOW=${parts[3]}; SS=${parts[4]}; IMG=${parts[5]}
    CFG_NAME="${HOW}_ss${SS}_thr0.1_imgthr${IMG}_both"
    OUT=$OUT_BASE/ours_sd14_${PACK_KEY}pack/$DS/$CFG_NAME
    N_IMGS=$(ls -1 "$OUT"/*.png 2>/dev/null | wc -l)
    if [ "$N_IMGS" -ge "$EXP" ]; then echo "[GPU $GPU][skip] ours $PACK_KEY $DS $CFG_NAME ($N_IMGS/$EXP)"; continue; fi
    mkdir -p "$OUT"
    PACK=${PACK_PATH[$PACK_KEY]}
    echo "[GPU $GPU][run] ours $PACK_KEY $DS $CFG_NAME exp=$EXP"
    cd $REPO/SafeGen
    CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
      --prompts "$PROMPTS" --outdir "$OUT" \
      --probe_mode both --cas_threshold 0.6 \
      --safety_scale $SS --attn_threshold 0.1 --img_attn_threshold $IMG \
      --how_mode $HOW --family_guidance --family_config "$PACK" \
      >> "$LOGDIR/nudity_ours_${PACK_KEY}_${DS}_${CFG_NAME}_g${GPU}.log" 2>&1
  fi
done
echo "[GPU $GPU/slot $SLOT] nudity full done at $(date)"
