#!/usr/bin/env python3
"""Vanilla SD1.4 baseline runner for NFE ablation.
Generates 60 imgs from prompts file at given num_inference_steps.
NO safegen, NO SAFREE — pure SD1.4 pipe(prompt)."""
import argparse, os, re
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline


def slugify(t, n=50):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", t.strip())[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"[baseline-SD1.4] steps={args.steps} seed={args.seed} cfg={args.cfg_scale}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=args.device)
    for i, prompt in enumerate(prompts):
        if i < args.start_idx: continue
        gen.manual_seed(args.seed + i)
        try:
            img = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg_scale,
                generator=gen,
                height=512, width=512,
            ).images[0]
        except Exception as e:
            print(f"[err] prompt {i}: {e}")
            continue
        name = slugify(prompt)
        img.save(f"{args.outdir}/{i:04d}_00_{name}.png")
        if (i+1) % 10 == 0:
            print(f"  baseline {i+1}/{len(prompts)} done")

    print(f"Done! {len(prompts)} prompts processed.")


if __name__ == "__main__":
    main()
