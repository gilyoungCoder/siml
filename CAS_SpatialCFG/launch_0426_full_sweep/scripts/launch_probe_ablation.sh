#!/bin/bash
# Probe-mode ablation: text vs image vs both, all 7 i2p concepts.
# attn_threshold = img_attn_threshold = 0.1 fixed (user spec).
# Other args from concept paper-best (cas, safety_scale, target_concepts).
# n_img_tokens=4, hybrid mode, family_guidance ON.
# Args: $1=GPU, $2=concept
set -uo pipefail
GPU=${1:-0}
CONCEPT=${2:-violence}
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
OUTBASE=$BASE/outputs/phase_probe_ablation
LOG=$BASE/logs/probe_ab_g${GPU}_${CONCEPT}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE
> $LOG

JQ () { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); v=d
for k in sys.argv[2:]: v=v[k]
print(v)" "$@"; }
JQARR () { $PY -c "import json,sys; d=json.load(open(sys.argv[1]))
for x in d[sys.argv[2]]: print(x)" "$@"; }

declare -A CONCEPT_ARGS_DIR=(
  [violence]=violence
  [shocking]=shocking
  [illegal_activity]=illegal
  [harassment]=harassment
  [hate]=hate
  [self-harm]=self-harm
  [sexual]=sexual
)

ARGS_DIR=${CONCEPT_ARGS_DIR[$CONCEPT]}
ARGS=$BASE/paper_results/single/$ARGS_DIR/args.json
[ -f "$ARGS" ] || { echo "missing args.json $CONCEPT"; exit 1; }

PROMPTS=$(JQ "$ARGS" prompts)
HOW=$(JQ "$ARGS" how_mode)
CAS=$(JQ "$ARGS" cas_threshold)
SS=$(JQ "$ARGS" safety_scale)
PACK=$(JQ "$ARGS" family_config)
TC_ARR=()
while IFS= read -r line; do TC_ARR+=("$line"); done < <(JQARR "$ARGS" target_concepts)

echo "[$(date)] $CONCEPT cas=$CAS ss=$SS pack=$(basename $PACK)" | tee -a $LOG

for MODE in text image both; do
  OUTDIR=$OUTBASE/${CONCEPT}_${MODE}
  mkdir -p $OUTDIR
  EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  if [ "$EXISTING" -ge 60 ]; then
    echo "[skip] $CONCEPT $MODE ($EXISTING/60)" | tee -a $LOG; continue
  fi
  echo "[$(date +%H:%M:%S)] [$CONCEPT $MODE] gen start" | tee -a $LOG
  cd /mnt/home3/yhgil99/unlearning/SafeGen
  CUDA_VISIBLE_DEVICES=$GPU $PY -m safegen.generate_family \
    --prompts "$PROMPTS" \
    --outdir "$OUTDIR" \
    --family_guidance --family_config "$PACK" \
    --probe_mode $MODE \
    --how_mode "$HOW" \
    --cas_threshold "$CAS" \
    --safety_scale "$SS" \
    --attn_threshold 0.1 \
    --img_attn_threshold 0.1 \
    --n_img_tokens 4 \
    --steps 50 --seed 42 --cfg_scale 7.5 \
    --target_concepts "${TC_ARR[@]}" >> $LOG 2>&1
  final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] [$CONCEPT $MODE] DONE imgs=$final" | tee -a $LOG
done
echo "[$(date)] $CONCEPT done" | tee -a $LOG
