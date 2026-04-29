#!/bin/bash
# Image-count saturation across all 7 i2p concepts.
# Each (concept, K) cell uses concept paper-best config, only swapping --family_config to per-K pack
# and --probe_mode image. K ∈ {1,2,4,8,12,16}. n_img_tokens=4 constant.
# 3-way split:
#  HALF=0: violence + illegal_activity (12 cells)
#  HALF=1: shocking + harassment (12 cells)
#  HALF=2: hate + self-harm + sexual_K12 (13 cells)
set -uo pipefail
GPU=${1:-0}
HALF=${2:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=$REPO/.conda/envs/sdd_copy/bin/python3.10
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
  KS=(1 2 4 8 12 16)
elif [ "$HALF" = "1" ]; then
  CONCEPTS=(shocking harassment)
  KS=(1 2 4 8 12 16)
elif [ "$HALF" = "2" ]; then
  CONCEPTS=(hate self-harm)
  KS=(1 2 4 8 12 16)
else
  echo "unknown HALF=$HALF" | tee -a $LOG; exit 1
fi

for C in "${CONCEPTS[@]}"; do
  ARGS_DIR=${CONCEPT_ARGS_DIR[$C]}
  ARGS=$BASE/paper_results/single/$ARGS_DIR/args.json
  if [ ! -f "$ARGS" ]; then echo "[err] no args.json for $C ($ARGS)" | tee -a $LOG; continue; fi
  for K in "${KS[@]}"; do
    PACK=$PACK_BASE/$C/clip_grouped_K${K}.pt
    if [ ! -f "$PACK" ]; then echo "[err] no pack $C K=$K ($PACK)" | tee -a $LOG; continue; fi
    OUTDIR=$OUTBASE/${C}_K${K}
    mkdir -p $OUTDIR
    EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 60 ]; then
        echo "[skip] $C K=$K ($EXISTING/60)" | tee -a $LOG
        continue
    fi
    echo "[$(date +%H:%M:%S)] [$C K=$K] gen start" | tee -a $LOG
    cd $REPO/SafeGen
    $PY -c "
import json, subprocess
a = json.load(open(\"$ARGS\"))
cmd = [\"env\", \"CUDA_VISIBLE_DEVICES=$GPU\", \"$PY\", \"-m\", \"safegen.generate_family\",
       \"--prompts\", a[\"prompts\"], \"--outdir\", \"$OUTDIR\",
       \"--family_guidance\", \"--family_config\", \"$PACK\",
       \"--probe_mode\", \"image\",
       \"--how_mode\", a[\"how_mode\"],
       \"--cas_threshold\", str(a[\"cas_threshold\"]),
       \"--safety_scale\", str(a[\"safety_scale\"]),
       \"--attn_threshold\", str(a[\"attn_threshold\"]),
       \"--img_attn_threshold\", str(a[\"img_attn_threshold\"]),
       \"--n_img_tokens\", \"4\",
       \"--steps\", \"50\", \"--seed\", \"42\", \"--cfg_scale\", \"7.5\",
       \"--target_concepts\", *a[\"target_concepts\"]]
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
    final=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [$C K=$K] DONE imgs=$final" | tee -a $LOG
  done
done

# Special: sexual K=12 only (other K values already done at phase_img_sat_all/)
if [ "$HALF" = "2" ]; then
  C=sexual
  ARGS=$BASE/paper_results/single/sexual/args.json
  K=12
  PACK=$PACK_BASE/$C/clip_grouped_K${K}.pt
  OUTDIR=$OUTBASE/${C}_K${K}
  mkdir -p $OUTDIR
  EXISTING=$(ls $OUTDIR/*.png 2>/dev/null | wc -l)
  if [ "$EXISTING" -lt 60 ]; then
    echo "[$(date +%H:%M:%S)] [sexual K=12] gen start" | tee -a $LOG
    cd $REPO/SafeGen
    $PY -c "
import json, subprocess
a = json.load(open(\"$ARGS\"))
cmd = [\"env\", \"CUDA_VISIBLE_DEVICES=$GPU\", \"$PY\", \"-m\", \"safegen.generate_family\",
       \"--prompts\", a[\"prompts\"], \"--outdir\", \"$OUTDIR\",
       \"--family_guidance\", \"--family_config\", \"$PACK\",
       \"--probe_mode\", \"image\",
       \"--how_mode\", a[\"how_mode\"],
       \"--cas_threshold\", str(a[\"cas_threshold\"]),
       \"--safety_scale\", str(a[\"safety_scale\"]),
       \"--attn_threshold\", str(a[\"attn_threshold\"]),
       \"--img_attn_threshold\", str(a[\"img_attn_threshold\"]),
       \"--n_img_tokens\", \"4\",
       \"--steps\", \"50\", \"--seed\", \"42\", \"--cfg_scale\", \"7.5\",
       \"--target_concepts\", *a[\"target_concepts\"]]
subprocess.run(cmd, check=False)
" >> $LOG 2>&1
    echo "[$(date +%H:%M:%S)] [sexual K=12] DONE" | tee -a $LOG
  fi
fi

echo "[$(date)] half=$HALF done" | tee -a $LOG
