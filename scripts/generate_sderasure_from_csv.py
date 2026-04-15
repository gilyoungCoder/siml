#!/usr/bin/env python3
"""
SDErasure: Generate images from a fine-tuned UNet using prompts from a CSV file.
Wrapper around the SDErasure generate pipeline for batch CSV-based generation.
"""
import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler


def load_prompts_from_csv(csv_path):
    """Extract prompts from CSV with various column formats."""
    df = pd.read_csv(csv_path)

    # Try different column names
    for col in ["adv_prompt", "sensitive prompt", "prompt"]:
        if col in df.columns:
            prompts = df[col].dropna().tolist()
            # Try to get seeds and case numbers too
            seeds = df["evaluation_seed"].tolist() if "evaluation_seed" in df.columns else [42] * len(prompts)
            case_nums = df["case_number"].tolist() if "case_number" in df.columns else list(range(len(prompts)))
            return prompts, seeds, case_nums

    raise ValueError(f"No prompt column found in {csv_path}. Columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    # Output dirs matching Safe_Denoiser format
    all_dir = os.path.join(args.output_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    # Load prompts
    prompts, seeds, case_nums = load_prompts_from_csv(args.csv)
    print(f"Loaded {len(prompts)} prompts from {args.csv}")

    # Load pipeline with fine-tuned UNet
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    if os.path.isdir(args.unet_dir):
        print(f"Loading fine-tuned UNet from: {args.unet_dir}")
        unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=torch.float16).to(args.device)
        pipe.unet = unet
    else:
        print(f"WARNING: UNet dir not found: {args.unet_dir}, using original")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    gen = torch.Generator(device=args.device)

    for i, (prompt, seed, case_num) in enumerate(tqdm(zip(prompts, seeds, case_nums), total=len(prompts), desc="Generating")):
        out_path = os.path.join(all_dir, f"{case_num}.png")
        if os.path.exists(out_path):
            continue

        try:
            seed_val = int(seed) if pd.notna(seed) else args.seed
        except (ValueError, TypeError):
            seed_val = args.seed

        try:
            img = pipe(
                str(prompt),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                generator=gen.manual_seed(seed_val),
            ).images[0]
            img.save(out_path)
        except Exception as e:
            print(f"Error on case {case_num}: {e}")
            continue

    total = len([f for f in os.listdir(all_dir) if f.endswith('.png')])
    print(f"Done. Total images in {all_dir}: {total}")


if __name__ == "__main__":
    main()
