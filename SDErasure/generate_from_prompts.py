#!/usr/bin/env python3
"""
Generate images from a prompt file using SDErasure fine-tuned model.
One image per prompt (standard for Ring-A-Bell / I2P evaluation).

Usage:
  python generate_from_prompts.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir ./outputs/sderasure_nudity/unet \
    --prompt_file ../prompts/nudity_datasets/ringabell.txt \
    --output_dir ./outputs/sderasure_nudity/ringabell_images \
    --seed 42
"""

import os
import argparse

import torch
from tqdm import tqdm
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)


def main():
    args = parse_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Replace UNet if fine-tuned weights provided
    if args.unet_dir and os.path.isdir(args.unet_dir):
        print(f"Loading fine-tuned UNet: {args.unet_dir}")
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_dir, torch_dtype=torch.float16
        ).to(device)
        pipe.unet = unet
    else:
        print("Using original model (no fine-tuned UNet)")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # Read prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # Generate
    os.makedirs(args.output_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        img = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.image_size,
            width=args.image_size,
            generator=generator,
        ).images[0]
        img.save(os.path.join(args.output_dir, f"{idx:06d}.png"))

    print(f"\nGenerated {len(prompts)} images → {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--unet_dir", type=str, default=None)
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    main()
