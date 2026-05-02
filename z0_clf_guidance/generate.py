#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate images with z0 classifier guidance for Stable Diffusion 1.4.

Supports two modes via --space:

[Latent space] (--space latent)
  Chain rule: zt -> Tweedie -> z0_hat -> Classifier(4ch) -> log_prob
  grad = d(log_prob)/d(zt) = d(log_prob)/d(z0_hat) * d(z0_hat)/d(zt)
                            = d(log_prob)/d(z0_hat) * 1/sqrt(alpha_bar)

[Image space] (--space image)
  Chain rule: zt -> Tweedie -> z0_hat -> VAE.decode -> x0_hat -> Classifier(3ch) -> log_prob
  grad = d(log_prob)/d(zt) = d(log_prob)/d(x0_hat) * d(x0_hat)/d(z0_hat) * d(z0_hat)/d(zt)
  VAE decode is IN the gradient path. More expensive but uses standard image classifiers.

Both modes:
  noise_pred is DETACHED (no gradient through UNet).
  Score-based guidance: adjusted_score = score + scale * grad -> scheduler.step
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
        description="Image generation with z0 classifier guidance"
    )
    parser.add_argument("ckpt_path", type=str, help="SD model path or HF model id")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="File with prompts (.txt: one per line, .csv: reads 'sensitive prompt' column)")
    parser.add_argument("--output_dir", type=str, default="output_img/z0_guided")
    parser.add_argument("--nsamples", type=int, default=1,
                        help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    # Classifier guidance
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Path to trained z0 classifier checkpoint")
    parser.add_argument("--architecture", type=str, default="resnet18",
                        choices=["resnet18", "vit_b"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "image"],
                        help="Classifier space: 'latent' (4ch z0) or 'image' (3ch x0, VAE decode in grad path)")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step to start applying guidance (0-indexed)")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance (0=non-people, 1=non-nude, 2=nude)")
    # Guidance mode
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm", "paired"],
                        help="'target': max log p(target). 'safe_minus_harm': aggregate. 'paired': argmax(harm)->safe pair")
    parser.add_argument("--safe_classes", type=int, nargs="+", default=None,
                        help="Safe class indices for safe_minus_harm mode (e.g. 0 1)")
    parser.add_argument("--harm_classes", type=int, nargs="+", default=None,
                        help="Harm class indices for safe_minus_harm mode (e.g. 2)")
    # Spatial guidance
    parser.add_argument("--spatial_guidance", action="store_true",
                        help="(Legacy) Enable spatial masking. Use --spatial_mode instead.")
    parser.add_argument("--spatial_mode", type=str, default="none",
                        choices=["none", "gradcam", "attention", "attention_gradcam"],
                        help="Spatial masking mode: "
                             "none=no masking, "
                             "gradcam=gradient magnitude, "
                             "attention=cross-attention heatmaps, "
                             "attention_gradcam=product of both")
    parser.add_argument("--spatial_threshold", type=float, default=0.3,
                        help="Threshold for spatial mask (0-1)")
    parser.add_argument("--spatial_soft", action="store_true",
                        help="Use continuous spatial mask instead of binary")
    parser.add_argument("--grad_wrt_z0", action="store_true",
                        help="Compute gradient w.r.t. z0_hat instead of z_t (removes 1/sqrt(alpha) amplification)")
    # Attention-based spatial guidance
    parser.add_argument("--harmful_keywords", type=str, nargs="+", default=None,
                        help="Keywords whose cross-attention maps define harmful regions "
                             "(e.g. 'nude' 'weapon' 'blood'). Required for attention modes.")
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=None,
                        help="UNet resolutions to hook for attention extraction "
                             "(e.g. 16 32 64). Default: all resolutions.")
    # Example-aware gating (EACG)
    parser.add_argument("--prototype_path", type=str, default=None,
                        help="Path to .pt prototype file for example-aware gating")
    parser.add_argument("--eacg_harm_class", type=int, default=2,
                        help="Harm class index for prototype gate (default: 2=nude)")
    parser.add_argument("--eacg_safe_class", type=int, default=1,
                        help="Safe class index for prototype gate (default: 1=clothed)")
    parser.add_argument("--eacg_tau", type=float, default=0.0,
                        help="Gate threshold (0=neutral boundary)")
    parser.add_argument("--eacg_kappa", type=float, default=0.1,
                        help="Gate temperature (smaller=sharper)")
    # GradCAM & spatial schedule
    parser.add_argument("--gradcam_layer", type=str, default="layer2",
                        help="ResNet layer for GradCAM (layer1/layer2/layer3/layer4)")
    parser.add_argument("--harmful_stats_path", type=str, default=None,
                        help="Path to .pt file with harmful GradCAM stats for CDF normalization")
    parser.add_argument("--harm_ratio", type=float, default=1.0,
                        help="Alpha for safe - alpha*harm in safe_minus_harm mode")
    parser.add_argument("--threshold_schedule", type=str, default="constant",
                        choices=["constant", "cosine"],
                        help="Threshold schedule: constant or cosine annealing")
    # CSV filtering (for I2P)
    parser.add_argument("--csv_prompt_column", type=str, default="prompt",
                        help="CSV column name for prompts (default: 'prompt')")
    parser.add_argument("--csv_filter_column", type=str, default=None,
                        help="CSV column to filter on (e.g. 'categories')")
    parser.add_argument("--csv_filter_value", type=str, default=None,
                        help="Value to match in filter column (substring match)")
    return parser.parse_args()


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
            gate_val = result.get("gate_val", 1.0)
            gate_score = result.get("gate_score", 0.0)
            if step % 10 == 0:
                msg = f"  step={step}, t={timestep}, monitor={val:.4f}"
                if mask_ratio < 1.0:
                    msg += f", mask={mask_ratio:.2%}"
                if gate_val < 0.99:
                    msg += f", gate={gate_val:.3f}(s={gate_score:.3f})"
                print(msg)

    return callback_kwargs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load SD pipeline (use DDIM scheduler for classifier guidance compatibility)
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
                # Filter by category if specified
                if args.csv_filter_column and args.csv_filter_value:
                    cell = row.get(args.csv_filter_column, "")
                    if args.csv_filter_value not in cell:
                        continue
                # Try common prompt column names
                prompt = row.get(args.csv_prompt_column,
                         row.get("sensitive prompt",
                         row.get("prompt", "")))
                if prompt.strip():
                    prompts.append(prompt.strip())
        filter_msg = ""
        if args.csv_filter_column:
            filter_msg = f" (filtered: {args.csv_filter_column}={args.csv_filter_value})"
        print(f"Loaded {len(prompts)} prompts from CSV{filter_msg}: {args.prompt_file}")
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup z0 classifier guidance
    model_config = {
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "space": args.space,
        "guidance_mode": args.guidance_mode,
        "safe_classes": args.safe_classes,
        "harm_classes": args.harm_classes,
        "spatial_guidance": args.spatial_guidance,
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
        print(f"\n[{idx + 1}/{len(prompts)}] {prompt}")

        # Set prompt for attention-based guidance (computes token indices)
        guidance_model.set_prompt(prompt, pipe.tokenizer)

        cb = partial(
            callback_on_step_end,
            guidance_model=guidance_model,
            guidance_scale=args.guidance_scale,
            guidance_start_step=args.guidance_start_step,
            target_class=args.target_class,
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

        for si, image in enumerate(output.images):
            fname = f"prompt_{idx + 1:04d}_sample_{si + 1}.png"
            save_image(image, fname, root=args.output_dir)

    print(f"\nDone. {len(prompts)} prompts x {args.nsamples} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
