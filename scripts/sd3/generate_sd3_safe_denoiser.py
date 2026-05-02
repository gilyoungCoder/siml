#!/usr/bin/env python
"""
Safe_Denoiser on SD3: Wrapper that calls Safe_Denoiser's existing SD3 pipeline.

Safe_Denoiser uses embedding projection + token masking for concept erasure.
This script provides a simplified CLI that creates the JSON config and invokes
the existing run_nudity_sdv3.py.
"""

import os, sys, json, csv, re
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Import Safe_Denoiser's SD3 pipelines
SAFE_DENOISER_DIR = os.path.join(os.path.dirname(__file__), "../../Safe_Denoiser")
sys.path.insert(0, SAFE_DENOISER_DIR)

from models.sdv3.sdv3_pipeline import StableDiffusion3Pipeline
from models.sdv3.safe_denoiser_pipeline import StableDiffusion3Pipeline as SaferDiffusion3Pipeline

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# Safe_Denoiser negative prompt space (from their code)
NEGATIVE_PROMPT_SPACE = [
    "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
    "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
    "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
    "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
]


def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts, seeds = [], []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            prompt_col = None
            for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']:
                if col in reader.fieldnames:
                    prompt_col = col
                    break
            seed_col = None
            for col in ['evaluation_seed', 'sd_seed', 'seed']:
                if col in reader.fieldnames:
                    seed_col = col
                    break
            for row in reader:
                prompts.append(row[prompt_col].strip())
                if seed_col and row.get(seed_col):
                    try:
                        seeds.append(int(row[seed_col]))
                    except (ValueError, TypeError):
                        seeds.append(None)
                else:
                    seeds.append(None)
        return prompts, seeds
    else:
        lines = [l.strip() for l in open(f) if l.strip()]
        return lines, [None] * len(lines)


def slugify(txt, maxlen=50):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", txt.strip())[:maxlen]


def main():
    p = ArgumentParser(description="Safe_Denoiser on SD3")
    p.add_argument("--prompts", required=True, help="CSV or TXT file with prompts")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--model_id", default=SD3_MODEL_ID)
    p.add_argument("--mode", choices=["std", "safree_neg_prompt"], default="safree_neg_prompt",
                   help="std=vanilla SD3, safree_neg_prompt=with negative prompt space")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=3.5)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--cpu_offload", action="store_true", default=True)
    p.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false")
    args = p.parse_args()

    device = torch.device(args.device)

    # Choose pipeline based on mode
    if args.mode == "safree_neg_prompt":
        pipeline_cls = SaferDiffusion3Pipeline
        negative_prompt = ", ".join(NEGATIVE_PROMPT_SPACE)
        print(f"Mode: Safe_Denoiser (safree_neg_prompt)")
    else:
        pipeline_cls = StableDiffusion3Pipeline
        negative_prompt = ""
        print(f"Mode: vanilla SD3")

    print(f"Loading SD3 pipeline from {args.model_id} ...")
    pipe = pipeline_cls.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled")
    else:
        pipe = pipe.to(device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)

    prompts, seeds = load_prompts(args.prompts)
    if args.end is not None:
        prompts = prompts[args.start:args.end]
        seeds = seeds[args.start:args.end]
    else:
        prompts = prompts[args.start:]
        seeds = seeds[args.start:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Safe_Denoiser-SD3: {len(prompts)} prompts, mode={args.mode}")
    print(f"  steps={args.steps}, cfg={args.cfg_scale}, res={args.resolution}")

    gen_device = "cpu" if args.cpu_offload else device
    gen = torch.Generator(device=gen_device)

    for i, prompt in enumerate(tqdm(prompts, desc=f"SafeDenoiser-SD3")):
        global_idx = args.start + i
        s = seeds[i] if seeds[i] is not None else args.seed + global_idx
        gen.manual_seed(s)

        img = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            height=args.resolution,
            width=args.resolution,
            generator=gen,
        ).images[0]

        name = slugify(prompt)
        img.save(str(outdir / f"{global_idx:04d}_00_{name}.png"))

    stats = {
        "method": f"safe_denoiser_sd3_{args.mode}",
        "model_id": args.model_id,
        "args": vars(args),
        "total_images": len(prompts),
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"Done! {len(prompts)} images saved to {outdir}")


if __name__ == "__main__":
    main()
