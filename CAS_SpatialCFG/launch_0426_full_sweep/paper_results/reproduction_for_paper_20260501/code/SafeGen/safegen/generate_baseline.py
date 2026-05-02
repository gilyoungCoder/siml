#!/usr/bin/env python
"""
Baseline: Standard Stable Diffusion v1.4 generation without safety guidance.

Usage:
    python -m safegen.generate_baseline \
        --prompts prompts/i2p_sexual.txt \
        --outdir outputs/baseline/sexual
"""

import json
import random
import csv
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


def load_prompts(filepath):
    fp = Path(filepath)
    if fp.suffix == ".csv":
        prompts = []
        with open(fp) as f:
            reader = csv.DictReader(f)
            col = next(
                (c for c in ["sensitive prompt", "adv_prompt", "prompt", "target_prompt", "text"]
                 if c in reader.fieldnames), None)
            for row in reader:
                prompts.append(row[col].strip())
        return prompts
    return [l.strip() for l in open(fp) if l.strip()]


def main():
    p = ArgumentParser(description="SD v1.4 Baseline (no safety guidance)")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    prompts = prompts[args.start_idx:end]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline: {len(prompts)} prompts x {args.nsamples} samples")

    for i, prompt in enumerate(tqdm(prompts)):
        idx = args.start_idx + i
        for j in range(args.nsamples):
            seed = args.seed + idx * args.nsamples + j
            gen = torch.Generator(device).manual_seed(seed)
            img = pipe(
                prompt, num_inference_steps=args.steps,
                guidance_scale=args.cfg_scale, generator=gen,
            ).images[0]
            img.save(str(outdir / f"{idx:04d}_{j:02d}.png"))

    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)
    print(f"Done! {len(prompts) * args.nsamples} images at {outdir}")


if __name__ == "__main__":
    main()
