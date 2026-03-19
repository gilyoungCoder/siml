#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grid search over z0 classifier guidance hyperparameters.

Loads the SD pipeline and classifier ONCE, then iterates over all
parameter combinations. Each combination writes images to a unique
output directory.

Usage:
    python grid_search.py CompVis/stable-diffusion-v1-4 \\
        --prompt_file prompts.txt \\
        --classifier_ckpt ./ckpt.pth \\
        --output_root ./grid_search_output \\
        --guidance_scales 5.0 10.0 20.0 \\
        --spatial_modes none gradcam attention attention_gradcam \\
        --spatial_thresholds 0.2 0.3 0.5 \\
        --harmful_keywords nude weapon blood
"""

import csv
import itertools
import json
import os
import random
import time
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from PIL import Image

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import Z0GuidanceModel


def parse_args():
    parser = ArgumentParser(description="Grid search for z0 classifier guidance")

    # ── Fixed arguments (same across all combinations) ──
    parser.add_argument("ckpt_path", type=str,
                        help="SD model path or HF model id")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="grid_search_output")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="resnet18",
                        choices=["resnet18", "vit_b"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "image"])
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm", "paired"])
    parser.add_argument("--safe_classes", type=int, nargs="+", default=None)
    parser.add_argument("--harm_classes", type=int, nargs="+", default=None)
    parser.add_argument("--grad_wrt_z0", action="store_true")
    # EACG (fixed)
    parser.add_argument("--prototype_path", type=str, default=None)
    parser.add_argument("--eacg_harm_class", type=int, default=2)
    parser.add_argument("--eacg_safe_class", type=int, default=1)
    parser.add_argument("--eacg_tau", type=float, default=0.0)
    parser.add_argument("--eacg_kappa", type=float, default=0.1)
    # CSV filtering
    parser.add_argument("--csv_prompt_column", type=str, default="prompt")
    parser.add_argument("--csv_filter_column", type=str, default=None)
    parser.add_argument("--csv_filter_value", type=str, default=None)

    # ── Swept parameters ──
    parser.add_argument("--guidance_scales", type=float, nargs="+",
                        default=[5.0, 10.0, 20.0],
                        help="Classifier guidance scales to sweep")
    parser.add_argument("--spatial_modes", type=str, nargs="+",
                        default=["none", "gradcam"],
                        help="Spatial modes to sweep "
                             "(none, gradcam, attention, attention_gradcam)")
    parser.add_argument("--spatial_thresholds", type=float, nargs="+",
                        default=[0.3],
                        help="Spatial thresholds to sweep")
    parser.add_argument("--spatial_soft_options", type=int, nargs="+",
                        default=[0],
                        help="Spatial soft mask options: 0=binary, 1=soft")
    parser.add_argument("--guidance_start_steps", type=int, nargs="+",
                        default=[1],
                        help="Guidance start steps to sweep")
    parser.add_argument("--harmful_keywords", type=str, nargs="+", default=None,
                        help="Keywords for attention-based guidance")
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=None,
                        help="UNet resolutions for attention extraction")

    return parser.parse_args()


# ─────────────────── helpers ───────────────────


def make_tag(combo):
    """Create a filesystem-safe tag string from parameter combination."""
    parts = [f"gs{combo['guidance_scale']}"]
    parts.append(f"sm_{combo['spatial_mode']}")
    if combo["spatial_mode"] != "none":
        parts.append(f"st{combo['spatial_threshold']}")
        if combo["spatial_soft"]:
            parts.append("soft")
    if combo["guidance_start_step"] != 1:
        parts.append(f"start{combo['guidance_start_step']}")
    return "_".join(parts)


def load_prompts(args):
    """Load prompts from file."""
    if args.prompt_file.endswith(".csv"):
        prompts = []
        with open(args.prompt_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.csv_filter_column and args.csv_filter_value:
                    cell = row.get(args.csv_filter_column, "")
                    if args.csv_filter_value not in cell:
                        continue
                prompt = row.get(
                    args.csv_prompt_column,
                    row.get("sensitive prompt", row.get("prompt", "")),
                )
                if prompt.strip():
                    prompts.append(prompt.strip())
        filter_msg = ""
        if args.csv_filter_column:
            filter_msg = (f" (filtered: {args.csv_filter_column}"
                          f"={args.csv_filter_value})")
        print(f"Loaded {len(prompts)} prompts from CSV{filter_msg}")
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    return prompts


def save_image(image, filename, root):
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def callback_on_step_end(pipe, step, timestep, callback_kwargs,
                         guidance_model=None, guidance_scale=5.0,
                         guidance_start_step=1, target_class=1):
    """Pipeline callback: apply z0 classifier guidance at each step."""
    if step >= guidance_start_step:
        result = guidance_model.guidance(
            pipe, callback_kwargs, step, timestep,
            guidance_scale, target_class=target_class,
        )
        callback_kwargs["latents"] = result["latents"]

        monitor = result.get("differentiate_value", None)
        if monitor is not None:
            val = monitor.mean().item()
            mask_ratio = result.get("spatial_mask_ratio", 1.0)
            if step % 10 == 0:
                msg = f"  step={step}, t={timestep}, monitor={val:.4f}"
                if mask_ratio < 1.0:
                    msg += f", mask={mask_ratio:.2%}"
                print(msg)

    return callback_kwargs


# ─────────────────── main ───────────────────


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pipeline ONCE
    print("Loading SD pipeline...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load prompts ONCE
    prompts = load_prompts(args)

    # Build parameter grid
    grid = list(itertools.product(
        args.guidance_scales,
        args.spatial_modes,
        args.spatial_thresholds,
        args.spatial_soft_options,
        args.guidance_start_steps,
    ))

    print(f"\nGrid: {len(grid)} combinations")
    for i, (gs, sm, st, ss, gss) in enumerate(grid):
        print(f"  [{i + 1}] gs={gs}, mode={sm}, thresh={st}, "
              f"soft={bool(ss)}, start={gss}")

    total_t0 = time.time()

    for combo_idx, (gs, sm, st, ss, gss) in enumerate(grid):
        combo = {
            "guidance_scale": gs,
            "spatial_mode": sm,
            "spatial_threshold": st,
            "spatial_soft": bool(ss),
            "guidance_start_step": gss,
        }
        tag = make_tag(combo)
        output_dir = os.path.join(args.output_root, tag)

        # Skip if already done
        expected_images = len(prompts) * args.nsamples
        if os.path.exists(output_dir):
            existing = len([f for f in os.listdir(output_dir)
                            if f.endswith(".png")])
            if existing >= expected_images:
                print(f"\n[{combo_idx + 1}/{len(grid)}] SKIP (done): {tag}")
                continue

        print(f"\n{'=' * 60}")
        print(f"[{combo_idx + 1}/{len(grid)}] {tag}")
        print(f"{'=' * 60}")

        # Reproducible seeds per combo
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Build guidance model for this combination
        model_config = {
            "architecture": args.architecture,
            "num_classes": args.num_classes,
            "space": args.space,
            "guidance_mode": args.guidance_mode,
            "safe_classes": args.safe_classes,
            "harm_classes": args.harm_classes,
            "spatial_guidance": sm != "none",
            "spatial_mode": sm,
            "spatial_threshold": st,
            "spatial_soft": bool(ss),
            "grad_wrt_z0": args.grad_wrt_z0,
            "harmful_keywords": args.harmful_keywords or [],
            "attn_resolutions": args.attn_resolutions,
            "prototype_path": args.prototype_path,
            "eacg_harm_class": args.eacg_harm_class,
            "eacg_safe_class": args.eacg_safe_class,
            "eacg_tau": args.eacg_tau,
            "eacg_kappa": args.eacg_kappa,
        }

        guidance_model = Z0GuidanceModel(
            pipe, args.classifier_ckpt, model_config,
            target_class=args.target_class, device=device,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(
                {**combo, "tag": tag, "classifier_ckpt": args.classifier_ckpt,
                 "space": args.space, "guidance_mode": args.guidance_mode,
                 "cfg_scale": args.cfg_scale, "seed": args.seed},
                f, indent=2,
            )

        t0 = time.time()

        for idx, prompt in enumerate(prompts):
            print(f"  [{idx + 1}/{len(prompts)}] {prompt[:80]}...")

            guidance_model.set_prompt(prompt, pipe.tokenizer)

            cb = partial(
                callback_on_step_end,
                guidance_model=guidance_model,
                guidance_scale=gs,
                guidance_start_step=gss,
                target_class=args.target_class,
            )

            with torch.enable_grad():
                output = pipe(
                    prompt=prompt,
                    guidance_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=512, width=512,
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=[
                        "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
                    ],
                    num_images_per_prompt=args.nsamples,
                )

            for si, image in enumerate(output.images):
                fname = f"prompt_{idx + 1:04d}_sample_{si + 1}.png"
                save_image(image, fname, root=output_dir)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s -> {output_dir}")

        # Cleanup: restore original attention processors for next combo
        guidance_model.cleanup(pipe.unet)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"Grid search complete! {len(grid)} combinations x {len(prompts)} prompts")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Results in: {args.output_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
