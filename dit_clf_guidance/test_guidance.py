#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity test: sweep guidance scales on random noise for Pony V7 (AuraFlow).

For each seed, generates one baseline (no guidance) and one image per scale,
then saves a row comparison image with classifier probabilities.

Usage:
    python test_guidance.py \\
        --classifier_ckpt work_dirs/.../classifier.pth \\
        --scales 0 1 3 5 10 15 20
    python test_guidance.py --prompt "a person" --n 3
"""

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models.latent_classifier import LatentResNet18Classifier
from utils.auraflow_utils import load_auraflow_components, encode_prompt
from generate import guided_euler_sample

CLASS_NAMES = {0: "non-people", 1: "clothed", 2: "nude"}

VAE_SCALE = 0.13025


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="purplesmartai/pony-v7-base")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./test_guidance_output")
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0, 1, 3, 5, 10, 15, 20],
                        help="Guidance scales to sweep (0 = no guidance)")
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt (default: empty = unconditional)")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm"])
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--safe_classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--harm_classes", type=int, nargs="+", default=[2])
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def classify_image(classifier, vae, image, device):
    """Classify a PIL image by encoding to z0 and running classifier."""
    with torch.no_grad():
        img_t = torch.tensor(
            np.array(image.resize((512, 512))), dtype=torch.float32, device=device,
        ).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        z0 = vae.encode(img_t).latent_dist.mean * VAE_SCALE
        logits = classifier(z0)
        probs = F.softmax(logits, dim=-1)
    return probs[0].cpu().numpy()


def make_row_image(images, labels, sub_labels):
    """Create a horizontal row of images with labels."""
    w, h = 256, 256
    n = len(images)
    margin = 4
    canvas = Image.new("RGB", (n * (w + margin) - margin, h + 48), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13,
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11,
        )
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

    model_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32,
    }[args.mixed_precision]

    scales = sorted(args.scales)
    prompt_label = f'"{args.prompt}"' if args.prompt else "(empty)"

    print("=" * 60)
    print("GUIDANCE SCALE SWEEP TEST (Pony V7 / AuraFlow)")
    print("=" * 60)
    print(f"  Prompt:     {prompt_label}")
    print(f"  Mode:       {args.guidance_mode}")
    print(f"  Scales:     {scales}")
    print(f"  Seeds:      {args.n}")
    print(f"  Classifier: {args.classifier_ckpt}")
    print("=" * 60)

    # Load AuraFlow / Pony V7 components
    print("\nLoading AuraFlow / Pony V7 components...")
    components = load_auraflow_components(
        args.pretrained_model_name_or_path, device=device, dtype=model_dtype,
    )
    vae = components["vae"]
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    scheduler = components["scheduler"]

    # Encode prompt + negative
    print("Encoding prompts...")
    prompt_embeds = encode_prompt(
        tokenizer, text_encoder, args.prompt or "",
        device=device, max_sequence_length=args.max_sequence_length,
        dtype=model_dtype,
    )
    neg_embeds = encode_prompt(
        tokenizer, text_encoder, "",
        device=device, max_sequence_length=args.max_sequence_length,
        dtype=model_dtype,
    )

    # Load classifier
    print("Loading classifier...")
    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes, pretrained_backbone=False,
    ).to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_ckpt, map_location=device)
    )
    classifier.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

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
                print("  scale=0 (no guidance)")
            else:
                print(f"  scale={scale}")

            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.enable_grad():
                images = guided_euler_sample(
                    transformer=transformer,
                    vae=vae,
                    scheduler=scheduler,
                    classifier=classifier if scale > 0 else None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    num_steps=args.num_inference_steps,
                    cfg_scale=args.cfg_scale,
                    guidance_scale=scale,
                    target_class=args.target_class,
                    guidance_mode=args.guidance_mode,
                    safe_classes=args.safe_classes,
                    harm_classes=args.harm_classes,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    device=device,
                    model_dtype=model_dtype,
                    verbose=False,
                )

            img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)

            probs = classify_image(classifier, vae, img_pil, device)
            pred = CLASS_NAMES[probs.argmax()]

            print(
                f"    -> {pred}  "
                f"[np={probs[0]:.1%} cl={probs[1]:.1%} nu={probs[2]:.1%}]"
            )

            all_results.append({
                "seed": seed, "scale": scale, "pred": pred, "probs": probs,
            })

            # Save individual
            tag = "no_guide" if scale == 0 else f"gs{scale}"
            img_pil.save(os.path.join(args.output_dir, f"seed{seed}_{tag}.png"))

            row_images.append(img_pil)
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
            r = next(r for r in all_results
                     if r["seed"] == seed and r["scale"] == s)
            p = r["probs"]
            row_str += f"  {r['pred']:>4s}({p[1]:.0%}/{p[2]:.0%})"
        print(row_str)

    print(f"\n  Format: pred(clothed%/nude%)")
    print(f"\nImages: {args.output_dir}/")
    print("  seed*_row.png  - scale comparison per seed")


if __name__ == "__main__":
    main()
