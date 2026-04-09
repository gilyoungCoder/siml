#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
DATASET=${2:?dataset}
METHOD=${3:?method} # baseline|sld

export PYTHONNOUSERSITE=1
export MAX_INFER_BATCH_SIZE=${MAX_INFER_BATCH_SIZE:-4}
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
REPO=/mnt/home3/yhgil99/unlearning
GEN=$REPO/../guided2-safe-diffusion
VLD=$REPO/vlm
OUTBASE=$REPO/unlearning-baselines/official_rerun

case "$DATASET" in
  nudity_rab)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt; EC=nudity ;;
  nudity_p4dn) PF=$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv; EC=nudity ;;
  nudity_ud)   PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv; EC=nudity ;;
  nudity_mma)  PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv; EC=nudity ;;
  nudity_i2p)  PF=$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv; EC=nudity ;;
  violence)    PF=$REPO/SAFREE/datasets/i2p_categories/i2p_violence.csv; EC=violence ;;
  harassment)  PF=$REPO/SAFREE/datasets/i2p_categories/i2p_harassment.csv; EC=harassment ;;
  hate)        PF=$REPO/SAFREE/datasets/i2p_categories/i2p_hate.csv; EC=hate ;;
  shocking)    PF=$REPO/SAFREE/datasets/i2p_categories/i2p_shocking.csv; EC=shocking ;;
  illegal|illegal_activity) PF=$REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv; EC=illegal ; DATASET=illegal_activity ;;
  self_harm|selfharm) PF=$REPO/SAFREE/datasets/i2p_categories/i2p_self-harm.csv; EC=self_harm ; DATASET=self_harm ;;
  *) echo "unknown dataset: $DATASET"; exit 2 ;;
esac

case "$METHOD" in
  baseline) ODIR=$OUTBASE/baseline_guided/$DATASET; EXTRA=() ;;
  sld) ODIR=$OUTBASE/sld_guided/$DATASET; EXTRA=(--pipeline_type sld --pipeline_config max) ;;
  *) echo "unknown method: $METHOD"; exit 2 ;;
esac

if [[ "$PF" == *.csv ]]; then
  CLEAN=/tmp/${DATASET}_guided_clean.csv
  PF_IN="$PF" CLEAN_OUT="$CLEAN" python3 - <<'PY2'
import os, pandas as pd
src=os.environ['PF_IN']; dst=os.environ['CLEAN_OUT']
df=pd.read_csv(src)
col=None
for c in ['adv_prompt','sensitive prompt','prompt','target_prompt','text','Prompt','Text']:
    if c in df.columns:
        col=c; break
if col is None:
    col=df.columns[0]
out=pd.DataFrame({'prompt': df[col].astype(str)})
out=out[out['prompt'].notna()]
out=out[out['prompt'].str.lower()!='nan']
out=out[out['prompt'].str.strip()!='']
if 'evaluation_seed' in df.columns: out['evaluation_seed']=df.loc[out.index,'evaluation_seed']
if 'sd_seed' in df.columns and 'evaluation_seed' not in out.columns: out['evaluation_seed']=df.loc[out.index,'sd_seed']
out.to_csv(dst,index=False)
print(dst, len(out))
PY2
  PF="$CLEAN"
fi

if [ -f "$ODIR/results_qwen3_vl_${EC}.txt" ]; then
  echo "[SKIP] $METHOD $DATASET"
  exit 0
fi
mkdir -p "$ODIR"

cd "$GEN"
echo "[$(date +%H:%M)] GPU $GPU: $METHOD $DATASET"
CUDA_VISIBLE_DEVICES=$GPU "$P" generate.py \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --image_dir "$ODIR" \
  --prompt_path "$PF" \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --seed 42 \
  --use_fp16 \
  --overwrite \
  --device cuda:0 \
  "${EXTRA[@]}"

N=$(find "$ODIR" -maxdepth 1 -name '*.png' | wc -l)
if [ "$N" -ge 10 ]; then
  echo "[$(date +%H:%M)] Eval $METHOD $DATASET ($N imgs)"
  CUDA_VISIBLE_DEVICES=$GPU "$VLP" "$VLD/opensource_vlm_i2p_all.py" "$ODIR" "$EC" qwen
fi

echo "[$(date +%H:%M)] DONE: $METHOD $DATASET"
