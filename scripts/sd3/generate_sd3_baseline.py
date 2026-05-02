#!/usr/bin/env python
"""SD3 Baseline: Standard Stable Diffusion 3 generation without any safety guidance."""

import os, json, csv
from argparse import ArgumentParser
from pathlib import Path
import torch
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline


SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

# SD3 recommended defaults
SD3_DEFAULTS = dict(
    steps=28,
    cfg_scale=7.0,
    resolution=1024,
    seed=42,
    nsamples=1,
)


def load_prompts(f):
    """Load prompts from CSV or TXT, matching the same column priority as SD1.4 baseline."""
    f = Path(f)
    if f.suffix == ".csv":
        prompts, seeds = [], []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            # Column priority (matches project convention)
            prompt_col = None
            for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']:
                if col in reader.fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"No recognized prompt column in {f}. Columns: {reader.fieldnames}")

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


def main():
    p = ArgumentParser(description="SD3 Baseline Generator")
    p.add_argument("--prompts", required=True, help="CSV or TXT file with prompts")
    p.add_argument("--outdir", required=True, help="Output directory for images")
    p.add_argument("--model_id", default=SD3_MODEL_ID, help="HF model ID or local path")
    p.add_argument("--nsamples", type=int, default=SD3_DEFAULTS["nsamples"],
                   help="Number of images per prompt")
    p.add_argument("--steps", type=int, default=SD3_DEFAULTS["steps"],
                   help="Number of inference steps")
    p.add_argument("--cfg_scale", type=float, default=SD3_DEFAULTS["cfg_scale"],
                   help="Guidance scale")
    p.add_argument("--resolution", type=int, default=SD3_DEFAULTS["resolution"],
                   help="Image resolution (height=width)")
    p.add_argument("--seed", type=int, default=SD3_DEFAULTS["seed"],
                   help="Global seed (used when CSV has no per-prompt seed)")
    p.add_argument("--device", default="cuda", help="Device")
    p.add_argument("--start", type=int, default=0, help="Start prompt index")
    p.add_argument("--end", type=int, default=None, help="End prompt index")
    p.add_argument("--cpu_offload", action="store_true", default=True,
                   help="Enable model CPU offload to save VRAM (default: on)")
    p.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false")
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"Loading SD3 from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled (saves VRAM)")
    else:
        pipe = pipe.to(device)

    prompts, seeds = load_prompts(args.prompts)

    # Apply start/end slicing
    if args.end is not None:
        prompts = prompts[args.start:args.end]
        seeds = seeds[args.start:args.end]
    else:
        prompts = prompts[args.start:]
        seeds = seeds[args.start:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"SD3 Baseline: {len(prompts)} prompts x {args.nsamples} samples")
    print(f"  steps={args.steps}, cfg={args.cfg_scale}, res={args.resolution}, seed={args.seed}")

    # When using CPU offload, generator must be on CPU
    gen_device = "cpu" if args.cpu_offload else device
    gen = torch.Generator(device=gen_device)

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        global_idx = args.start + i
        for j in range(args.nsamples):
            # Use per-prompt seed from CSV if available, otherwise compute from global seed
            if seeds[i] is not None:
                s = seeds[i] + j
            else:
                s = args.seed + global_idx * args.nsamples + j

            gen.manual_seed(s)

            img = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=args.steps,
                guidance_scale=args.cfg_scale,
                height=args.resolution,
                width=args.resolution,
                generator=gen,
            ).images[0]

            # File naming: consistent with project convention
            name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            img.save(str(outdir / f"{global_idx:04d}_{j:02d}_{name}.png"))

    # Save metadata
    stats = {
        "method": "sd3_baseline",
        "model_id": args.model_id,
        "args": vars(args),
        "total_images": len(prompts) * args.nsamples,
    }
    json.dump(stats, open(outdir / "stats.json", "w"), indent=2)
    print(f"Done! {len(prompts) * args.nsamples} images saved to {outdir}")


if __name__ == "__main__":
    main()
