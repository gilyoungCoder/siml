#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity test: sweep guidance scales on random noise.

For each seed, generates one baseline (no guidance) and one image per scale,
then saves a row comparison image.

Usage:
    python test_guidance.py
    python test_guidance.py --scales 1 2 5 10 20
    python test_guidance.py --prompt "a person" --n 3
"""

import os
import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import Z0GuidanceModel
from models.latent_classifier import LatentResNet18Classifier


CLASS_NAMES = {0: "non-people", 1: "clothed", 2: "nude"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth")
    parser.add_argument("--output_dir", type=str, default="./test_guidance_output")
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0, 1, 3, 5, 10, 15, 20],
                        help="Guidance scales to sweep (0 = no guidance)")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt (default: empty = unconditional)")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm"])
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--safe_classes", type=int, nargs="+", default=[1])
    parser.add_argument("--harm_classes", type=int, nargs="+", default=[2])
    return parser.parse_args()


def classify_image(classifier, vae, image, device):
    with torch.no_grad():
        img_t = torch.tensor(
            np.array(image.resize((512, 512))), dtype=torch.float32, device=device
        ).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        z0 = vae.encode(img_t).latent_dist.mean * 0.18215
        logits = classifier(z0)
        probs = F.softmax(logits, dim=-1)
    return probs[0].cpu().numpy()


def guidance_callback(pipe, step, timestep, callback_kwargs,
                      guidance_model=None, guidance_scale=10.0, target_class=1):
    result = guidance_model.guidance(
        pipe, callback_kwargs, step, timestep,
        guidance_scale, target_class=target_class,
    )
    callback_kwargs["latents"] = result["latents"]
    if step % 10 == 0:
        monitor = result.get("differentiate_value", None)
        if monitor is not None:
            print(f"      step={step}, t={timestep}, gap={monitor.mean().item():.4f}")
    return callback_kwargs


def generate(pipe, prompt, seed, num_steps, cfg_scale, callback=None):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.enable_grad():
        output = pipe(
            prompt=prompt,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            height=512, width=512,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=[
                "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
            ],
            num_images_per_prompt=1,
        )
    return output.images[0]


def make_row_image(images, labels, sub_labels):
    """Create a horizontal row of images with labels."""
    w, h = 256, 256
    n = len(images)
    margin = 4
    canvas = Image.new("RGB", (n * (w + margin) - margin, h + 48), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_sm = font

    for i, (img, label, sub) in enumerate(zip(images, labels, sub_labels)):
        x = i * (w + margin)
        canvas.paste(img.resize((w, h)), (x, 0))
        draw.text((x + 4, h + 2), label, fill=(0, 0, 0), font=font)
        draw.text((x + 4, h + 20), sub, fill=(100, 100, 100), font=font_sm)

    return canvas


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scales = sorted(args.scales)
    prompt_label = f'"{args.prompt}"' if args.prompt else "(empty)"

    print("=" * 60)
    print("GUIDANCE SCALE SWEEP TEST")
    print("=" * 60)
    print(f"  Prompt:   {prompt_label}")
    print(f"  Mode:     {args.guidance_mode}")
    print(f"  Scales:   {scales}")
    print(f"  Seeds:    {args.n}")
    print(f"  Classifier: {args.classifier_ckpt}")
    print("=" * 60)

    # Load pipeline
    print("\nLoading SD pipeline...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    print("Loading classifier...")
    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes, pretrained_backbone=False
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()

    # Setup guidance model
    model_config = {
        "architecture": "resnet18",
        "num_classes": args.num_classes,
        "space": "latent",
        "guidance_mode": args.guidance_mode,
        "safe_classes": args.safe_classes,
        "harm_classes": args.harm_classes,
        "spatial_mode": "none",
        "spatial_threshold": 0.3,
        "spatial_soft": False,
        "grad_wrt_z0": False,
    }
    guidance_model = Z0GuidanceModel(
        pipe, args.classifier_ckpt, model_config,
        target_class=args.target_class, device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []  # (seed, scale) -> probs

    for i in range(args.n):
        seed = args.seed + i
        print(f"\n{'=' * 60}")
        print(f"Seed {seed}  ({i + 1}/{args.n})")
        print("=" * 60)

        row_images = []
        row_labels = []
        row_subs = []

        for scale in scales:
            if scale == 0:
                print(f"  scale=0 (no guidance)")
                img = generate(pipe, args.prompt, seed,
                               args.num_inference_steps, args.cfg_scale, callback=None)
            else:
                print(f"  scale={scale}")
                guidance_model.set_prompt(args.prompt, pipe.tokenizer)
                cb = partial(guidance_callback, guidance_model=guidance_model,
                             guidance_scale=scale, target_class=args.target_class)
                img = generate(pipe, args.prompt, seed,
                               args.num_inference_steps, args.cfg_scale, callback=cb)

            probs = classify_image(classifier, pipe.vae, img, device)
            pred = CLASS_NAMES[probs.argmax()]

            print(f"    -> {pred}  [np={probs[0]:.1%} cl={probs[1]:.1%} nu={probs[2]:.1%}]")

            all_results.append({"seed": seed, "scale": scale, "pred": pred, "probs": probs})

            # Save individual
            tag = "no_guide" if scale == 0 else f"gs{scale}"
            img.save(os.path.join(args.output_dir, f"seed{seed}_{tag}.png"))

            row_images.append(img)
            label = "no guide" if scale == 0 else f"gs={scale}"
            row_labels.append(f"{label}: {pred}")
            row_subs.append(f"cl={probs[1]:.0%} nu={probs[2]:.0%}")

        # Save row comparison
        row = make_row_image(row_images, row_labels, row_subs)
        row.save(os.path.join(args.output_dir, f"seed{seed}_row.png"))

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Header
    header = f"  {'seed':>6s}"
    for s in scales:
        label = "none" if s == 0 else f"gs={s}"
        header += f"  {label:>12s}"
    print(header)
    print("  " + "-" * (8 + 14 * len(scales)))

    for i in range(args.n):
        seed = args.seed + i
        row_str = f"  {seed:>6d}"
        for s in scales:
            r = next(r for r in all_results if r["seed"] == seed and r["scale"] == s)
            p = r["probs"]
            row_str += f"  {r['pred']:>4s}({p[1]:.0%}/{p[2]:.0%})"
        print(row_str)

    print(f"\n  Format: pred(clothed%/nude%)")
    print(f"\nImages: {args.output_dir}/")
    print(f"  seed*_row.png  - scale comparison per seed")


if __name__ == "__main__":
    main()
