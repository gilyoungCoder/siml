#!/bin/bash
# Plain SD1.4 baseline COCO 10k chunk — no safety guidance, vanilla diffusers.
# Usage: bash baseline_coco_chunk.sh <gpu> <start> <end>
set -uo pipefail
GPU=$1; START=$2; END=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0424_coco10k/baseline_sd14
PROMPTS=$REPO/CAS_SpatialCFG/prompts/coco_10k.txt
LOGDIR=$REPO/logs/launch_0424_coco10k
LOG=$LOGDIR/baseline_g${GPU}_${START}_${END}.log
mkdir -p "$OUTDIR" "$LOGDIR"

CUDA_VISIBLE_DEVICES=$GPU $PY -c "
import os, sys, torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

prompts = [l.strip() for l in open('$PROMPTS') if l.strip()]
start, end = $START, $END
if end < 0 or end > len(prompts): end = len(prompts)
prompts = prompts[start:end]
print(f'Generating {len(prompts)} prompts (idx {start}-{end})', flush=True)

pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32, safety_checker=None).to('cuda')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

gen = torch.Generator('cuda').manual_seed(42)
for i, p in enumerate(prompts):
    idx = start + i
    out_name = os.path.join('$OUTDIR', f'{idx:04d}_00.png')
    if os.path.exists(out_name):
        continue
    try:
        img = pipe(p, num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0]
        img.save(out_name)
    except Exception as e:
        print(f'[{idx}] failed: {e}', flush=True)
    if i % 50 == 0:
        print(f'  {idx}/{end} done', flush=True)
print('DONE', flush=True)
" >> "$LOG" 2>&1 || echo "[g$GPU] FAILED" | tee -a "$LOG"
echo "[$(date)] [g$GPU] baseline chunk $START-$END done" | tee -a "$LOG"
