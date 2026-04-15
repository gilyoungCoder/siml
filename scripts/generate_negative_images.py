#!/usr/bin/env python3
"""
Generate negative reference images from I2P sexual prompts using vanilla SD v1.4.
These are used by Safe_Denoiser and SGF as repellency reference data.
"""
import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="I2P sexual CSV file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=600, help="Max images to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check existing images
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])
    if existing >= args.max_images:
        print(f"Already have {existing} images in {args.output_dir}, skipping.")
        return

    # Load prompts
    df = pd.read_csv(args.csv)
    if "prompt" in df.columns:
        prompts = df["prompt"].dropna().tolist()
    elif "sensitive prompt" in df.columns:
        prompts = df["sensitive prompt"].dropna().tolist()
    else:
        raise ValueError(f"No prompt column found. Columns: {list(df.columns)}")

    prompts = prompts[:args.max_images]
    print(f"Generating {len(prompts)} images from {args.csv}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        out_path = os.path.join(args.output_dir, f"prompt_{i:04d}_sample_1.png")
        if os.path.exists(out_path):
            continue
        try:
            img = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            ).images[0]
            img.save(out_path)
        except Exception as e:
            print(f"Error on prompt {i}: {e}")
            continue

    total = len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])
    print(f"Done. Total images: {total}")


if __name__ == "__main__":
    main()
