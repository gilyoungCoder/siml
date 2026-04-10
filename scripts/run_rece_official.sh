#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?gpu}
DATASET=${2:?dataset}

export PYTHONNOUSERSITE=1
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
REPO=/mnt/home3/yhgil99/unlearning
RDIR=$REPO/unlearning-baselines/RECE
VLD=$REPO/vlm
OUTBASE=$REPO/unlearning-baselines/official_rerun/rece_official

case "$DATASET" in
  nudity_rab)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.csv; EC=nudity; CKPT=$RDIR/ckpts/nudity_ep2.pt ;;
  nudity_p4dn) PF=$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv; EC=nudity; CKPT=$RDIR/ckpts/nudity_ep2.pt ;;
  nudity_ud)   PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv; EC=nudity; CKPT=$RDIR/ckpts/nudity_ep2.pt ;;
  nudity_mma)  PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv; EC=nudity; CKPT=$RDIR/ckpts/nudity_ep2.pt ;;
  nudity_i2p)  PF=$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv; EC=nudity; CKPT=$RDIR/ckpts/nudity_ep2.pt ;;
  violence)    PF=$REPO/SAFREE/datasets/i2p_categories/i2p_violence.csv; EC=violence; CKPT=$RDIR/ckpts/unsafe_ep1.pt ;;
  harassment)  PF=$REPO/SAFREE/datasets/i2p_categories/i2p_harassment.csv; EC=harassment; CKPT=$RDIR/ckpts/unsafe_ep1.pt ;;
  hate)        PF=$REPO/SAFREE/datasets/i2p_categories/i2p_hate.csv; EC=hate; CKPT=$RDIR/ckpts/unsafe_ep1.pt ;;
  shocking)    PF=$REPO/SAFREE/datasets/i2p_categories/i2p_shocking.csv; EC=shocking; CKPT=$RDIR/ckpts/unsafe_ep1.pt ;;
  illegal|illegal_activity) PF=$REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv; EC=illegal; CKPT=$RDIR/ckpts/unsafe_ep1.pt; DATASET=illegal_activity ;;
  self_harm|selfharm) PF=$REPO/SAFREE/datasets/i2p_categories/i2p_self-harm.csv; EC=self_harm; CKPT=$RDIR/ckpts/unsafe_ep1.pt; DATASET=self_harm ;;
  *) echo "unknown dataset: $DATASET"; exit 2 ;;
esac

ODIR=$OUTBASE/$DATASET
if [ -f "$ODIR/results_qwen3_vl_${EC}.txt" ]; then
  echo "[SKIP] RECE $DATASET"
  exit 0
fi
mkdir -p "$OUTBASE"

if [ ! -f "$PF" ] && [ "$DATASET" = nudity_rab ]; then
  python3 - <<'PY2'
from pathlib import Path
src=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.txt')
dst=Path('/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.csv')
if not dst.exists():
    lines=[x.strip() for x in src.read_text().splitlines() if x.strip()]
    with dst.open('w') as f:
        f.write('prompt\n')
        for line in lines:
            f.write('"'+line.replace('"','""')+'"\n')
PY2
fi

cd "$RDIR"
echo "[$(date +%H:%M)] GPU $GPU: RECE $DATASET"
PYTHONPATH="$RDIR:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU PF="$PF" CKPT="$CKPT" ODIR="$ODIR" "$P" - <<'PY3'
import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from execs.generate_images import generate_images

pf = os.environ['PF']
ckpt = os.environ['CKPT']
outdir = os.environ['ODIR']
df = pd.read_csv(pf)
model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
if ckpt.endswith('pt'):
    try:
        model.unet.load_state_dict(torch.load(ckpt, map_location='cpu'))
    except Exception:
        ck = torch.load(ckpt, map_location='cpu')
        model.unet.load_state_dict(ck['state_dict'], strict=False)
generate_images(model, df, outdir, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=50, num_samples=1)
PY3

if [ -d "$ODIR/imgs" ]; then
  N=$(find "$ODIR/imgs" -maxdepth 1 -name '*.png' | wc -l)
  if [ "$N" -ge 10 ]; then
    echo "[$(date +%H:%M)] Eval RECE $DATASET ($N imgs)"
    CUDA_VISIBLE_DEVICES=$GPU "$VLP" "$VLD/opensource_vlm_i2p_all.py" "$ODIR/imgs" "$EC" qwen
  fi
fi

echo "[$(date +%H:%M)] DONE: RECE $DATASET"
