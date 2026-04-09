#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

GPU=${1:-0}
CONCEPT=${2:-nudity_rab}
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
I2P=$REPO/SAFREE/datasets/i2p_categories
OUTBASE=$REPO/unlearning-baselines/outputs/sld_official

case $CONCEPT in
  nudity_rab)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt; EC=nudity ;;
  nudity_p4dn) PF=$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv; EC=nudity ;;
  nudity_ud)   PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv; EC=nudity ;;
  nudity_mma)  PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv; EC=nudity ;;
  nudity_i2p)  PF=$I2P/i2p_sexual.csv; EC=nudity ;;
  violence)    PF=$I2P/i2p_violence.csv; EC=violence ;;
  harassment)  PF=$I2P/i2p_harassment.csv; EC=harassment ;;
  hate)        PF=$I2P/i2p_hate.csv; EC=hate ;;
  shocking)    PF=$I2P/i2p_shocking.csv; EC=shocking ;;
  illegal)     PF=$I2P/i2p_illegal_activity.csv; EC=illegal ;;
  self_harm)   PF=$I2P/i2p_self-harm.csv; EC=self_harm ;;
esac

ODIR=$OUTBASE/$CONCEPT
[ -f "$ODIR/categories_qwen3_vl_${EC}.json" ] && echo "[SKIP] $CONCEPT" && exit 0
mkdir -p "$ODIR"

N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -lt 10 ]; then
  echo "[$(date +%H:%M)] GPU $GPU: SLD-Max $CONCEPT"
  CUDA_VISIBLE_DEVICES=$GPU $P -c "
import torch, os, pandas as pd
from diffusers.pipelines.stable_diffusion_safe import StableDiffusionPipelineSafe, SafetyConfig

pf = '$PF'
outdir = '$ODIR'

if pf.endswith('.csv'):
    df = pd.read_csv(pf)
    col = 'adv_prompt' if 'adv_prompt' in df.columns else ('prompt' if 'prompt' in df.columns else df.columns[0])
    prompts = [str(p) for p in df[col].dropna().tolist() if str(p).strip()]
else:
    with open(pf) as f:
        prompts = [l.strip() for l in f if l.strip()]

print(f'SLD-Max: {len(prompts)} prompts -> {outdir}')
pipe = StableDiffusionPipelineSafe.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16, safety_checker=None).to('cuda')
cfg = SafetyConfig.MAX

for i, p in enumerate(prompts):
    op = os.path.join(outdir, f'{i:04d}_00.png')
    if os.path.exists(op): continue
    try:
        img = pipe(p, num_inference_steps=50, guidance_scale=7.5, generator=torch.Generator('cuda').manual_seed(42), **cfg).images[0]
        img.save(op)
    except: pass
    if (i+1)%100==0: print(f'  {i+1}/{len(prompts)}')
print(f'Done! {len([x for x in os.listdir(outdir) if x.endswith(\".png\")])} imgs')
" 2>&1 | tail -5
fi

N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -ge 10 ] && [ ! -f "$ODIR/categories_qwen3_vl_${EC}.json" ]; then
  echo "[$(date +%H:%M)] Eval SLD $CONCEPT ($N imgs)"
  CUDA_VISIBLE_DEVICES=$GPU $VLP $VLD/opensource_vlm_i2p_all.py "$ODIR" "$EC" qwen 2>&1 | tail -1
fi
echo "[$(date +%H:%M)] DONE: SLD $CONCEPT"
