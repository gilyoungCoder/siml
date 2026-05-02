#!/usr/bin/env python3
"""SDXL-Lightning 4-step image generator (vanilla, no classifier guidance).
Generates images for human eval alignment.

Usage:
    python sdxl_lightning_gen.py --prompts <file.txt> --outdir <dir> --start_idx N --end_idx M [--steps 4] [--seed 42]
"""
import os, argparse
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=4, help="SDXL-Lightning distilled steps (1/2/4/8 official; paper uses 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=0.0, help="Lightning requires cfg=0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{args.steps}step_unet.safetensors"

    print(f"[SDXL-Lightning] base={base} steps={args.steps}", flush=True)
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    end = min(end, len(prompts))
    chunk = prompts[args.start_idx:end]
    print(f"Generating {len(chunk)} prompts (idx {args.start_idx}-{end})", flush=True)

    gen = torch.Generator("cuda").manual_seed(args.seed)
    for i, p in enumerate(chunk):
        idx = args.start_idx + i
        out = os.path.join(args.outdir, f"{idx:05d}_00.png")
        if os.path.exists(out):
            continue
        try:
            img = pipe(p, num_inference_steps=args.steps, guidance_scale=args.cfg, generator=gen).images[0]
            img.save(out)
        except Exception as e:
            print(f"[{idx}] failed: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{idx+1}/{end}] done", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
