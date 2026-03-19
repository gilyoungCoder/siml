#!/usr/bin/env python3
"""
Generate training images from RingaBell 'Full' prompts using vanilla SD 1.4.
29 prompts × 52 seeds = 1508 images (~1500).
"""

import argparse
import csv
import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_csv", type=str,
                        default="ringabell_split/ringabell_train_full.csv")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/home/yhgil99/dataset/ringabell_nude_train")
    parser.add_argument("--model_id", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_images_per_prompt", type=int, default=52,
                        help="Number of images per prompt (52 × 29 = 1508 ≈ 1500)")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="Start prompt index (inclusive)")
    parser.add_argument("--prompt_end", type=int, default=-1,
                        help="End prompt index (exclusive), -1 for all")
    args = parser.parse_args()

    # Load prompts
    all_prompts = []
    with open(args.prompt_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_prompts.append(row["sensitive prompt"])
    print(f"Loaded {len(all_prompts)} total prompts")

    # Slice prompts for this worker
    end = args.prompt_end if args.prompt_end > 0 else len(all_prompts)
    prompt_indices = list(range(args.prompt_start, end))
    prompts = [all_prompts[i] for i in prompt_indices]
    print(f"This worker: prompts {args.prompt_start}-{end-1} ({len(prompts)} prompts)")

    total_images = len(prompts) * args.num_images_per_prompt
    print(f"Will generate {total_images} images ({args.num_images_per_prompt} per prompt)")

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print(f"Loading model: {args.model_id}")
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    # Generate
    generated = 0
    skipped = 0
    for local_idx, (p_idx, prompt) in enumerate(zip(prompt_indices, prompts)):
        seeds = list(range(args.start_seed, args.start_seed + args.num_images_per_prompt))

        for batch_start in range(0, len(seeds), args.batch_size):
            batch_seeds = seeds[batch_start:batch_start + args.batch_size]

            # Check which ones already exist
            todo_seeds = []
            todo_prompts = []
            for s in batch_seeds:
                fname = f"p{p_idx:03d}_s{s:04d}.png"
                if not (output_dir / fname).exists():
                    todo_seeds.append(s)
                    todo_prompts.append(prompt)
                else:
                    skipped += 1

            if not todo_seeds:
                continue

            # Generate batch
            generators = [torch.Generator(device=args.device).manual_seed(s) for s in todo_seeds]
            images = pipe(
                todo_prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                generator=generators,
                height=args.resolution,
                width=args.resolution,
            ).images

            for img, s in zip(images, todo_seeds):
                fname = f"p{p_idx:03d}_s{s:04d}.png"
                img.save(output_dir / fname)
                generated += 1

        done = (local_idx + 1) * args.num_images_per_prompt
        print(f"[{local_idx+1}/{len(prompts)}] prompt {p_idx}: generated {generated}, skipped {skipped}, total {done}/{total_images}")

    print(f"\nDone! Generated: {generated}, Skipped: {skipped}")
    print(f"Output: {output_dir}")
    print(f"Total images: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
