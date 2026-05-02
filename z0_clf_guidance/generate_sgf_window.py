#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate images with z0 classifier guidance + SGF-style time window.

Combines spatial classifier guidance with the critical time window insight
from Safety-Guided Flow (SGF): guidance is only applied during early denoising
steps (e.g., timestep 1000->800) where global structure is determined.

Key difference from generate.py:
  - --guidance_start_t / --guidance_end_t: timestep-based window (like SGF)
  - Guidance is applied ONLY when guidance_end_t <= timestep <= guidance_start_t
"""

import csv
import os
import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
from PIL import Image

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import Z0GuidanceModel


def parse_args():
    parser = ArgumentParser(
        description="Image generation with z0 classifier guidance + SGF time window"
    )
    parser.add_argument("ckpt_path", type=str, help="SD model path or HF model id")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_img/cg_sgf_window")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Classifier guidance
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "image"])
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--target_class", type=int, default=1)

    # SGF-style time window (timestep-based)
    parser.add_argument("--guidance_start_t", type=int, default=1000,
                        help="Start applying guidance at this timestep (high=early)")
    parser.add_argument("--guidance_end_t", type=int, default=800,
                        help="Stop applying guidance at this timestep (low=later)")

    # Guidance mode
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm", "paired"])
    parser.add_argument("--safe_classes", type=int, nargs="+", default=None)
    parser.add_argument("--harm_classes", type=int, nargs="+", default=None)
    parser.add_argument("--harm_ratio", type=float, default=1.0)

    # Spatial guidance
    parser.add_argument("--spatial_mode", type=str, default="none",
                        choices=["none", "gradcam", "attention", "attention_gradcam"])
    parser.add_argument("--spatial_threshold", type=float, default=0.3)
    parser.add_argument("--spatial_soft", action="store_true")
    parser.add_argument("--grad_wrt_z0", action="store_true")
    parser.add_argument("--gradcam_layer", type=str, default="layer2")
    parser.add_argument("--harmful_stats_path", type=str, default=None)
    parser.add_argument("--threshold_schedule", type=str, default="constant",
                        choices=["constant", "cosine"])

    # Attention-based spatial guidance
    parser.add_argument("--harmful_keywords", type=str, nargs="+", default=None)
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=None)

    # Example-aware gating
    parser.add_argument("--prototype_path", type=str, default=None)
    parser.add_argument("--eacg_harm_class", type=int, default=2)
    parser.add_argument("--eacg_safe_class", type=int, default=1)
    parser.add_argument("--eacg_tau", type=float, default=0.0)
    parser.add_argument("--eacg_kappa", type=float, default=0.1)

    # CSV options
    parser.add_argument("--csv_prompt_column", type=str, default="prompt")
    parser.add_argument("--csv_filter_column", type=str, default=None)
    parser.add_argument("--csv_filter_value", type=str, default=None)

    return parser.parse_args()


def save_image(image, filename, root):
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def callback_on_step_end(pipe, step, timestep, callback_kwargs,
                         guidance_model=None, guidance_scale=10.0,
                         target_class=1,
                         guidance_start_t=1000, guidance_end_t=800):
    """
    Pipeline callback with SGF-style time window.
    Guidance is applied ONLY when guidance_end_t <= timestep <= guidance_start_t.
    """
    t_val = timestep.item() if torch.is_tensor(timestep) else timestep

    # SGF critical window: only apply in early denoising steps
    if guidance_end_t <= t_val <= guidance_start_t:
        result = guidance_model.guidance(
            pipe, callback_kwargs, step, timestep,
            guidance_scale, target_class=target_class,
        )
        callback_kwargs["latents"] = result["latents"]

        monitor = result.get("differentiate_value", None)
        if monitor is not None:
            val = monitor.mean().item()
            mask_ratio = result.get("spatial_mask_ratio", 1.0)
            gate_val = result.get("gate_val", 1.0)
            gate_score = result.get("gate_score", 0.0)
            msg = f"  step={step}, t={t_val}, monitor={val:.4f}"
            if mask_ratio < 1.0:
                msg += f", mask={mask_ratio:.2%}"
            if gate_val < 0.99:
                msg += f", gate={gate_val:.3f}(s={gate_score:.3f})"
            print(msg)
    else:
        if step % 10 == 0:
            print(f"  step={step}, t={t_val} [outside window, no guidance]")

    return callback_kwargs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"=== Spatial CG + SGF Time Window ===")
    print(f"Guidance window: t=[{args.guidance_end_t}, {args.guidance_start_t}]")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial mode: {args.spatial_mode}")
    print(f"Seed: {args.seed}")

    # Load SD pipeline
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load prompts
    if args.prompt_file.endswith(".csv"):
        prompts = []
        with open(args.prompt_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.csv_filter_column and args.csv_filter_value:
                    cell = row.get(args.csv_filter_column, "")
                    if args.csv_filter_value not in cell:
                        continue
                prompt = row.get(args.csv_prompt_column,
                         row.get("sensitive prompt",
                         row.get("prompt", "")))
                if prompt.strip():
                    prompts.append(prompt.strip())
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup z0 classifier guidance
    model_config = {
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "space": args.space,
        "guidance_mode": args.guidance_mode,
        "safe_classes": args.safe_classes,
        "harm_classes": args.harm_classes,
        "spatial_mode": args.spatial_mode,
        "spatial_threshold": args.spatial_threshold,
        "spatial_soft": args.spatial_soft,
        "grad_wrt_z0": args.grad_wrt_z0,
        "harmful_keywords": args.harmful_keywords or [],
        "attn_resolutions": args.attn_resolutions,
        "prototype_path": args.prototype_path,
        "eacg_harm_class": args.eacg_harm_class,
        "eacg_safe_class": args.eacg_safe_class,
        "eacg_tau": args.eacg_tau,
        "eacg_kappa": args.eacg_kappa,
        "gradcam_layer": args.gradcam_layer,
        "harmful_stats_path": args.harmful_stats_path,
        "harm_ratio": args.harm_ratio,
        "threshold_schedule": args.threshold_schedule,
    }
    guidance_model = Z0GuidanceModel(
        pipe,
        args.classifier_ckpt,
        model_config,
        target_class=args.target_class,
        device=device,
    )

    # Generate images
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx}/{len(prompts)}] {prompt}")

        guidance_model.set_prompt(prompt, pipe.tokenizer)

        # Reset seed per prompt (same as SGF: seed=42 for all)
        torch.manual_seed(args.seed)

        cb = partial(
            callback_on_step_end,
            guidance_model=guidance_model,
            guidance_scale=args.guidance_scale,
            target_class=args.target_class,
            guidance_start_t=args.guidance_start_t,
            guidance_end_t=args.guidance_end_t,
        )

        with torch.enable_grad():
            output = pipe(
                prompt=prompt,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=512,
                width=512,
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=[
                    "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
                ],
                num_images_per_prompt=args.nsamples,
            )

        # Save with same naming as SGF: {idx}_nudity.png
        for si, image in enumerate(output.images):
            fname = f"{idx}_nudity.png"
            save_image(image, fname, root=args.output_dir)

    print(f"\nDone. {len(prompts)} prompts -> {args.output_dir}")


if __name__ == "__main__":
    main()
