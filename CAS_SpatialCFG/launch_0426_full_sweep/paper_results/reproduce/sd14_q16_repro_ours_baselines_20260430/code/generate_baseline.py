#!/usr/bin/env python
"""Baseline: Standard SD v1.4 generation without any safety guidance."""

import os, json, random, csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            for col in ['sensitive prompt', 'adv_prompt', 'prompt', 'target_prompt', 'text']:
                if col in reader.fieldnames:
                    prompt_col = col; break
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    return [l.strip() for l in open(f) if l.strip()]


def main():
    p = ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    prompts = load_prompts(args.prompts)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline: {len(prompts)} prompts × {args.nsamples} samples")

    for i, prompt in enumerate(tqdm(prompts)):
        for j in range(args.nsamples):
            seed = args.seed + i * args.nsamples + j
            gen = torch.Generator(device).manual_seed(seed)
            img = pipe(prompt, num_inference_steps=args.steps,
                       guidance_scale=args.cfg_scale, generator=gen).images[0]
            name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            img.save(str(outdir / f"{i:04d}_{j:02d}_{name}.png"))

    json.dump({"method": "baseline_sdv14", "args": vars(args), "total": len(prompts) * args.nsamples},
              open(outdir / "stats.json", "w"), indent=2)
    print(f"Done! {len(prompts) * args.nsamples} images at {outdir}")

if __name__ == "__main__":
    main()
