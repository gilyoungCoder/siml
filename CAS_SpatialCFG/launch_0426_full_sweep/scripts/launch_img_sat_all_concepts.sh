#!/bin/bash
# Image-count saturation across all 7 i2p concepts.
# Pure bash launcher — args.json values read via 'jq'.
set -uo pipefail
GPU=${1:-0}
HALF=${2:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=$REPO/.conda/envs/sdd_copy/bin/python3.10
JQ () { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); v=d
for k in sys.argv[2:]: v=v[k]
print(v)" "$@"; }
JQARR () { $PY -c "import json,sys; d=json.load(open(sys.argv[1]))
for x in d[sys.argv[2]]: print(x)" "$@"; }
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
PACK_BASE=$BASE/exemplars_K_per_concept
OUTBASE=$BASE/outputs/phase_img_sat_all
LOG=$BASE/logs/img_sat_all_g${GPU}_h${HALF}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE
> $LOG

declare -A CONCEPT_ARGS_DIR=(
  [violence]=violence
  [shocking]=shocking
  [illegal_activity]=illegal
  [harassment]=harassment
  [hate]=hate
  [self-harm]=self-harm
  [sexual]=sexual
)

if [ "$HALF" = "0" ]; then
  CONCEPTS=(violence illegal_activity)
elif [ "$HALF" = "1" ]; then
  CONCEPTS=(shocking harassment)
elif [ "$HALF" = "2" ]; then
  CONCEPTS=(hate self-harm)
else
  echo "unknown HALF=$HALF" | tee -a $LOG; exit 1
fi
KS=(1 2 4 8 12 16)

run_cell () {
  local C=$1 K=$2
  local ARGS_DIR=${CONCEPT_ARGS_DIR[$C]}
  local ARGS=$BASE/paper_results/single/$ARGS_DIR/args.json
  if [ ! -f "$ARGS" ]; then echo "[err] no args.json $C ($ARGS)" | tee -a $LOG; return; fi
  local PACK=$PACK_BASE/$C/clip_grouped_K${K}.pt
  if [ ! -f "$PACK" ]; then echo "[err] no pack $C K=$K" | tee -a $LOG; return; fi
  local OUTDIR=$OUTBASE/${C}_K${K}
  mkdir -p $OUTDIR
  local EXISTING
  EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  if [ "$EXISTING" -ge 60 ]; then
    echo "[skip] $C K=$K ($EXISTING/60)" | tee -a $LOG
    return
  fi
  # parse args.json fields
  local PROMPTS HOW CAS SS ATT IATT
  PROMPTS=$(JQ "$ARGS" prompts)
  HOW=$(JQ "$ARGS" how_mode)
  CAS=$(JQ "$ARGS" cas_threshold)
  SS=$(JQ "$ARGS" safety_scale)
  ATT=$(JQ "$ARGS" attn_threshold)
  IATT=$(JQ "$ARGS" img_attn_threshold)
  local TC_ARR=()
  while IFS= read -r line; do TC_ARR+=("$line"); done < <(JQARR "$ARGS" target_concepts)
  echo "[$(date +%H:%M:%S)] [$C K=$K] gen start (prompts=$(basename $PROMPTS), tc=${#TC_ARR[@]})" | tee -a $LOG
  cd $REPO/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --family_guidance --family_config "$PACK" \
    --probe_mode image \
    --how_mode "$HOW" \
    --cas_threshold "$CAS" \
    --safety_scale "$SS" \
    --attn_threshold "$ATT" \
    --img_attn_threshold "$IATT" \
    --n_img_tokens 4 \
    --steps 50 --seed 42 --cfg_scale 7.5 \
    --target_concepts "${TC_ARR[@]}" >> $LOG 2>&1
  local final
  final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] [$C K=$K] DONE imgs=$final" | tee -a $LOG
}

for C in "${CONCEPTS[@]}"; do
  for K in "${KS[@]}"; do
    run_cell "$C" "$K"
  done
done

# half=2: also run sexual K=12 (other K already copied from existing run)
if [ "$HALF" = "2" ]; then
  run_cell sexual 12
fi

echo "[$(date)] half=$HALF DONE" | tee -a $LOG
