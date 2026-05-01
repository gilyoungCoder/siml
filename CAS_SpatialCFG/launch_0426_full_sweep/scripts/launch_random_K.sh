#!/bin/bash
# Random-subsample multi-seed: 3 concepts × K∈{1,2} × 3 seeds = 18 cells
# Args: $1=GPU $2=WID $3=NWORK
set -uo pipefail
GPU=${1:-0}
WID=${2:-0}
NWORK=${3:-7}
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
PACK_BASE=$BASE/exemplars_K_random
OUTBASE=$BASE/outputs/phase_img_sat_random
LOG=$BASE/logs/random_K_g${GPU}_w${WID}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE
> $LOG

declare -A AD=([violence]=violence [sexual]=sexual [hate]=hate)

CELLS=()
for C in violence sexual hate; do
  for K in 1 2; do
    for SEED in 42 43 44; do
      CELLS+=("$C|$K|$SEED")
    done
  done
done

JQ () { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); v=d
for k in sys.argv[2:]: v=v[k]
print(v)" "$@"; }
JQARR () { $PY -c "import json,sys; d=json.load(open(sys.argv[1]))
for x in d[sys.argv[2]]: print(x)" "$@"; }

i=0
for entry in "${CELLS[@]}"; do
  if [ $((i % NWORK)) -eq $WID ]; then
    IFS='|' read -r C K SEED <<< "$entry"
    PACK=$PACK_BASE/$C/clip_grouped_K${K}_seed${SEED}.pt
    OUT=$OUTBASE/${C}_K${K}_seed${SEED}
    mkdir -p $OUT
    EXISTING=$(ls $OUT/*.png 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 60 ]; then
      echo "[$(date +%H:%M:%S)] [w$WID] SKIP ${C}_K${K}_s${SEED}" | tee -a $LOG
      i=$((i+1)); continue
    fi
    ARGS=$BASE/paper_results/single/${AD[$C]}/args.json
    PROMPTS=$(JQ "$ARGS" prompts)
    HOW=$(JQ "$ARGS" how_mode)
    CAS=$(JQ "$ARGS" cas_threshold)
    SS=$(JQ "$ARGS" safety_scale)
    ATT=$(JQ "$ARGS" attn_threshold)
    IATT=$(JQ "$ARGS" img_attn_threshold)
    TC_ARR=()
    while IFS= read -r line; do TC_ARR+=("$line"); done < <(JQARR "$ARGS" target_concepts)
    echo "[$(date +%H:%M:%S)] [w$WID] RUN ${C}_K${K}_s${SEED}" | tee -a $LOG
    cd /mnt/home3/yhgil99/unlearning/SafeGen
    CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
      --prompts "$PROMPTS" --outdir "$OUT" \
      --family_guidance --family_config "$PACK" \
      --probe_mode image --how_mode "$HOW" \
      --cas_threshold "$CAS" --safety_scale "$SS" \
      --attn_threshold "$ATT" --img_attn_threshold "$IATT" \
      --n_img_tokens 4 --steps 50 --seed 42 --cfg_scale 7.5 \
      --target_concepts "${TC_ARR[@]}" >> $LOG 2>&1
    final=$(ls $OUT/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [w$WID] DONE ${C}_K${K}_s${SEED} imgs=$final" | tee -a $LOG
  fi
  i=$((i+1))
done
echo "[$(date)] worker $WID done" | tee -a $LOG
