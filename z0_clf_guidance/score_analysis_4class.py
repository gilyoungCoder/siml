#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score analysis using 4-class UNet classifier (z_t, t input).

Generates images with vanilla SD (no guidance), records per-step
classifier scores for both COCO and ringabell.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image

# Import 4-class classifier from SoftDelete+CG
sys.path.insert(0, "/mnt/home/yhgil99/unlearning/SoftDelete+CG")
from geo_models.classifier.classifier import load_discriminator


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            column_priority = [
                'adv_prompt', 'sensitive prompt', 'prompt',
                'target_prompt', 'text', 'Prompt', 'Text'
            ]
            for col in column_priority:
                if col in fieldnames:
                    prompt_col = col
                    break
            else:
                raise ValueError(f"CSV has no recognizable prompt column. Available: {fieldnames}")
            print(f"[INFO] Using column '{prompt_col}' from {f}")
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    else:
        return [l.strip() for l in open(f) if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/"
                                "nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth")
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    if args.end_idx > 0:
        prompts = prompts[args.start_idx:args.end_idx]
    else:
        prompts = prompts[args.start_idx:]
    print(f"Loaded {len(prompts)} prompts [{args.start_idx}:{args.end_idx}]")

    # Load 4-class classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None, eval=True,
        channel=4, num_classes=4,
    ).to(device)
    classifier.eval()
    print(f"Loaded 4-class classifier: {args.classifier_ckpt}")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Class names
    class_names = ["benign", "clothed", "nude", "color"]

    all_stats = []

    for i, prompt in enumerate(prompts):
        prompt_idx = args.start_idx + i
        current_seed = args.seed + prompt_idx
        set_seed(current_seed)

        # Per-step score storage
        step_scores = []  # list of {step, t, probs, pred_class}

        def callback(pipe_obj, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]  # z_t: noisy latent

            with torch.no_grad():
                # 4-class classifier: input (z_t, t/1000)
                t_normalized = torch.tensor(
                    [timestep.item() / 1000.0], device=device, dtype=torch.float32
                )
                logits = classifier(latents, t_normalized)
                probs = F.softmax(logits, dim=-1)[0]  # (4,)

            step_scores.append({
                "step": step,
                "t": timestep.item(),
                "p_benign": probs[0].item(),
                "p_clothed": probs[1].item(),
                "p_nude": probs[2].item(),
                "p_color": probs[3].item(),
                "pred_class": probs.argmax().item(),
            })

            return callback_kwargs

        # Generate
        output = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg_scale,
            callback_on_step_end=callback,
        )

        # Save image
        safe_prompt = "".join(
            c if c.isalnum() or c in ' -_' else '_' for c in prompt
        )[:50].replace(' ', '_')
        img_filename = f"{prompt_idx:04d}_{safe_prompt}.png"
        img = output.images[0]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.resize((512, 512)).save(output_dir / img_filename)

        # Compute summary stats
        p_nude_history = [s["p_nude"] for s in step_scores]
        p_harm_history = [s["p_nude"] + s["p_color"] for s in step_scores]
        pred_classes = [s["pred_class"] for s in step_scores]

        max_p_nude = max(p_nude_history) if p_nude_history else 0
        max_p_harm = max(p_harm_history) if p_harm_history else 0
        n_nude_steps = sum(1 for c in pred_classes if c == 2)
        n_harm_steps = sum(1 for c in pred_classes if c in (2, 3))

        img_stats = {
            "prompt_idx": prompt_idx,
            "seed": current_seed,
            "prompt": prompt[:100],
            "filename": img_filename,
            "max_p_nude": round(max_p_nude, 4),
            "max_p_harm": round(max_p_harm, 4),
            "n_nude_steps": n_nude_steps,
            "n_harm_steps": n_harm_steps,
            "p_nude_history": [round(p, 4) for p in p_nude_history],
            "p_harm_history": [round(p, 4) for p in p_harm_history],
            "pred_class_history": pred_classes,
        }
        all_stats.append(img_stats)

        pred_str = "".join(str(c) for c in pred_classes)
        print(f"  [{prompt_idx:03d}] max_nude={max_p_nude:.3f} max_harm={max_p_harm:.3f} "
              f"nude_steps={n_nude_steps}/50 classes={pred_str[:20]}...")

    # Save
    summary = {
        "args": vars(args),
        "overall": {"total_images": len(all_stats)},
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(all_stats)} results to {output_dir}/generation_stats.json")


if __name__ == "__main__":
    main()
