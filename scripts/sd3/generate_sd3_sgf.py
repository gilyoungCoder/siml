#!/usr/bin/env python
"""
SGF (Safer Guided Diffusion) on SD3: Wrapper for SGF's repellency-based safety.

SGF uses latent-space repellency with temporal gating (neg_start/neg_end)
to push generated images away from unsafe reference embeddings.
This script provides a simplified CLI for our experiment pipeline.
"""

import os, sys, json, csv, re
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Import SGF's SD3 pipelines
SGF_DIR = os.path.join(os.path.dirname(__file__), "../../SGF/diversity_sdv3")
sys.path.insert(0, SGF_DIR)

from models.sdv3.sdv3_pipeline import StableDiffusion3Pipeline
from models.sdv3.safer_pipeline import StableDiffusion3Pipeline as SaferDiffusion3Pipeline

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"


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
    p = ArgumentParser(description="SGF on SD3 — repellency-based safety guidance")
    p.add_argument("--prompts", required=True, help="CSV or TXT file with prompts")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--model_id", default=SD3_MODEL_ID)
    p.add_argument("--mode", choices=["std", "sgf"], default="sgf",
                   help="std=vanilla, sgf=with repellency")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=3.5)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--cpu_offload", action="store_true", default=True)
    p.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false")
    # SGF repellency parameters
    p.add_argument("--neg_start", type=int, default=1000,
                   help="Timestep to start repellency (higher = earlier)")
    p.add_argument("--neg_end", type=int, default=800,
                   help="Timestep to end repellency")
    p.add_argument("--ref_data_dir", type=str, default=None,
                   help="Directory with reference images for repellency embedding")
    p.add_argument("--task_config", type=str, default=None,
                   help="YAML config for repellency (SGF native format)")
    args = p.parse_args()

    device = torch.device(args.device)

    # Choose pipeline
    if args.mode == "sgf":
        pipeline_cls = SaferDiffusion3Pipeline
        print(f"Mode: SGF (repellency)")
    else:
        pipeline_cls = StableDiffusion3Pipeline
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

    # Setup repellency processor if SGF mode with ref data
    repellency_processor = None
    neg_config = {}
    if args.mode == "sgf" and args.task_config is not None:
        # Use SGF's native repellency setup
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../SGF/diversity_sdv3"))
        from data.dataloader import get_dataset, get_dataloader, get_transform, get_all_imgs
        from repellency.repellency_methods_fast_sdv3 import get_repellency_method
        import yaml

        with open(args.task_config) as f:
            task_config = yaml.safe_load(f)

        data_config = task_config['data']
        transform = get_transform(**data_config)
        repellency_dataset = get_dataset(**data_config, transforms=transform)
        repellency_loader = get_dataloader(repellency_dataset, batch_size=200, num_workers=0, train=False)
        ref_imgs = get_all_imgs(repellency_loader).to(device)

        embed_fn = lambda x: pipe.vae.encode(x.to(torch.float16)).latent_dist.sample() * pipe.vae.config.scaling_factor

        rep_config = task_config['repellency']
        repellency_processor = get_repellency_method(
            rep_config['method'],
            ref_data=ref_imgs,
            embed_fn=embed_fn,
            n_embed=rep_config['n_embed'],
            scheduler=pipe.scheduler,
            **rep_config['params']
        )
        neg_config = {"neg_start": args.neg_start, "neg_end": args.neg_end}

    prompts, seeds = load_prompts(args.prompts)
    if args.end is not None:
        prompts = prompts[args.start:args.end]
        seeds = seeds[args.start:args.end]
    else:
        prompts = prompts[args.start:]
        seeds = seeds[args.start:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"SGF-SD3: {len(prompts)} prompts, mode={args.mode}")
    print(f"  steps={args.steps}, cfg={args.cfg_scale}, res={args.resolution}")

    gen_device = "cpu" if args.cpu_offload else device
    gen = torch.Generator(device=gen_device)

    for i, prompt in enumerate(tqdm(prompts, desc=f"SGF-SD3")):
        global_idx = args.start + i
        s = seeds[i] if seeds[i] is not None else args.seed + global_idx
        gen.manual_seed(s)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            height=args.resolution,
            width=args.resolution,
            generator=gen,
        )
        if repellency_processor is not None:
            pipe_kwargs["repellency_processor"] = repellency_processor
            pipe_kwargs.update(neg_config)

        img = pipe(**pipe_kwargs).images[0]

        name = slugify(prompt)
        img.save(str(outdir / f"{global_idx:04d}_00_{name}.png"))

    stats = {
        "method": f"sgf_sd3_{args.mode}",
        "model_id": args.model_id,
        "args": {k: str(v) for k, v in vars(args).items()},
        "total_images": len(prompts),
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"Done! {len(prompts)} images saved to {outdir}")


if __name__ == "__main__":
    main()
