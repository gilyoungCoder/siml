#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate images with z0 classifier guidance for Pony V7 (AuraFlow DiT).

Custom Euler flow matching sampler with classifier guidance:

  For each denoising step (sigma decreasing from ~1 to 0):
    1. AuraFlow forward (no_grad) -> velocity prediction (with CFG)
    2. x0_hat = x_t - sigma * v_pred  (identity Jacobian!)
    3. Classifier(x0_hat) -> log p(target) -> gradient w.r.t. x_t
    4. v_guided = v_pred + scale * grad
    5. Gradient clipping: ||guidance_term|| <= clip_ratio * ||v_pred||
    6. Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v_guided

Usage:
    # Baseline (no guidance)
    python generate.py --prompt_file prompts/country_nude_body.txt \\
        --output_dir output_img/baseline --guidance_scale 0

    # Guided generation
    python generate.py --prompt_file prompts/country_nude_body.txt \\
        --output_dir output_img/guided_s10 \\
        --classifier_ckpt work_dirs/.../classifier.pth \\
        --guidance_scale 10 --guidance_mode safe_minus_harm

    # Ring-a-Bell CSV prompts
    python generate.py --prompt_csv /path/to/nudity-ring-a-bell.csv \\
        --csv_column "sensitive prompt" --output_dir output_img/ringabell
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from models.latent_classifier import LatentResNet18Classifier
from utils.auraflow_utils import load_auraflow_components, encode_prompt, auraflow_forward

VAE_SCALE = 0.13025


def compute_guidance_value(logits, guidance_mode, target_class=1,
                           safe_classes=None, harm_classes=None):
    """
    Compute differentiable guidance value (to maximize).

    Modes:
      - "target": maximize log p(target_class)
      - "safe_minus_harm": maximize log(sum p(safe)) - log(sum p(harm))
    """
    log_probs = F.log_softmax(logits, dim=-1)

    if guidance_mode == "safe_minus_harm":
        safe_cls = safe_classes or [0, 1]
        harm_cls = harm_classes or [2]
        safe_lse = torch.logsumexp(log_probs[:, safe_cls], dim=-1)
        harm_lse = torch.logsumexp(log_probs[:, harm_cls], dim=-1)
        diff_val = (safe_lse - harm_lse).sum()
    else:
        diff_val = log_probs[:, target_class].sum()

    return diff_val


def guided_euler_sample(
    transformer,
    vae,
    scheduler,
    classifier,
    prompt_embeds,
    negative_prompt_embeds,
    num_steps=20,
    cfg_scale=3.5,
    guidance_scale=5.0,
    guidance_start_step=1,
    target_class=1,
    guidance_mode="target",
    safe_classes=None,
    harm_classes=None,
    grad_clip_ratio=0.3,
    height=1024,
    width=1024,
    generator=None,
    device="cuda",
    model_dtype=torch.bfloat16,
    verbose=True,
):
    """
    Custom Euler flow matching sampler with z0 classifier guidance for AuraFlow.

    Args:
        transformer: frozen AuraFlowTransformer2DModel
        vae: frozen AutoencoderKL (SDXL VAE, scaling_factor=0.13025)
        scheduler: FlowMatchEulerDiscreteScheduler
        classifier: trained LatentResNet18Classifier (or None for baseline)
        prompt_embeds: (1, seq_len, 2048) text embeddings (mask baked in)
        negative_prompt_embeds: (1, seq_len, 2048) for CFG
        guidance_scale: classifier guidance strength (0 = no guidance)

    Returns:
        images: (B, 3, H, W) tensor in [0, 1]
    """
    latent_h = height // 8
    latent_w = width // 8

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps  # [0, 1000] range, decreasing
    sigmas = scheduler.sigmas        # [0, 1] range, decreasing, ends with 0.0

    # Initialize from noise
    latents = torch.randn(
        1, 4, latent_h, latent_w, generator=generator, device=device,
    )
    # Flow matching: start from pure noise (sigma=1), no scaling needed

    # Prepare CFG prompt embeddings
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        full_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    use_guidance = (
        classifier is not None and guidance_scale > 0
    )

    for i, t in enumerate(timesteps):
        sigma = sigmas[i]
        prev_latents = latents.clone()

        # ---- CFG velocity prediction (no_grad) ----
        with torch.no_grad():
            if use_cfg:
                latent_input = torch.cat([latents] * 2).to(model_dtype)
                # Timestep normalized to [0, 1] inside auraflow_forward
                t_expand = t.expand(latent_input.shape[0])
                v_pred_all = auraflow_forward(
                    transformer, latent_input, t_expand, full_embeds,
                )
                v_pred_uncond, v_pred_text = v_pred_all.chunk(2)
                v_pred = (
                    v_pred_uncond
                    + cfg_scale * (v_pred_text - v_pred_uncond)
                )
            else:
                latent_input = latents.to(model_dtype)
                t_expand = t.unsqueeze(0)
                v_pred = auraflow_forward(
                    transformer, latent_input, t_expand, prompt_embeds,
                )
                v_pred_uncond = v_pred

        v_pred = v_pred.to(latents.dtype)
        v_pred_uncond = v_pred_uncond.to(latents.dtype)

        # ---- Classifier guidance ----
        if use_guidance and i >= guidance_start_step:
            x_t = prev_latents.clone().detach().float().requires_grad_(True)
            v_det = v_pred_uncond.detach().float()

            # Flow matching x0 prediction: x0_hat = x_t - sigma * v_pred
            # d(x0_hat)/d(x_t) = I (identity Jacobian — key advantage!)
            x0_hat = x_t - float(sigma) * v_det

            logits = classifier(x0_hat)
            diff_val = compute_guidance_value(
                logits, guidance_mode, target_class,
                safe_classes, harm_classes,
            )

            grad = torch.autograd.grad(diff_val, x_t)[0].detach()

            # Apply guidance directly to velocity (flow matching convention)
            guidance_term = guidance_scale * grad

            # Gradient clipping: limit guidance term to clip_ratio * ||v||
            gt_norm = guidance_term.norm()
            v_norm = v_pred.norm()
            max_guidance = v_norm * grad_clip_ratio
            clipped = False
            if gt_norm > max_guidance and gt_norm > 0:
                guidance_term = guidance_term * (max_guidance / gt_norm)
                clipped = True

            if verbose and i % 5 == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    prob_str = " ".join(
                        f"c{c}={probs[0,c]:.3f}" for c in range(logits.shape[1])
                    )
                print(
                    f"  step={i:3d}, sigma={float(sigma):.4f}, "
                    f"||gt||={gt_norm:.4f}, ||v||={v_norm:.4f}, "
                    f"clipped={clipped}, {prob_str}"
                )

            # Adjust velocity prediction
            v_pred = v_pred.float() + guidance_term
            v_pred = v_pred.to(prev_latents.dtype)

        # ---- Euler step via scheduler ----
        latents = scheduler.step(
            v_pred, t, prev_latents, return_dict=False,
        )[0]

    # ---- Decode ----
    with torch.no_grad():
        # VAE force_upcast: decode in float32 to avoid overflow
        latents_decode = latents.float() / VAE_SCALE
        images = vae.decode(latents_decode, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)

    return images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with z0 classifier guidance for Pony V7 (AuraFlow)"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="purplesmartai/pony-v7-base")
    # Prompts (one of --prompt_file or --prompt_csv required)
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Text file with one prompt per line")
    parser.add_argument("--prompt_csv", type=str, default=None,
                        help="CSV file (e.g. Ring-a-Bell nudity-ring-a-bell.csv)")
    parser.add_argument("--csv_column", type=str, default="sensitive prompt",
                        help="Column name to use from CSV (default: 'sensitive prompt')")
    parser.add_argument("--output_dir", type=str, default="output_img/guided")
    parser.add_argument("--nsamples", type=int, default=1,
                        help="Number of samples per prompt")
    # Sampling
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=3.5,
                        help="Text CFG scale (AuraFlow default: 3.5)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    # Classifier guidance
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Path to trained classifier. None = no guidance")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance strength (0 = off)")
    parser.add_argument("--guidance_start_step", type=int, default=1)
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm"])
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class (0=benign, 1=clothed, 2=nude)")
    parser.add_argument("--safe_classes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--harm_classes", type=int, nargs="+", default=[2])
    parser.add_argument("--grad_clip_ratio", type=float, default=0.3)
    # AuraFlow
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32,
    }[args.mixed_precision]

    # Load AuraFlow / Pony V7 components
    print("Loading AuraFlow / Pony V7 components...")
    components = load_auraflow_components(
        args.pretrained_model_name_or_path, device=device, dtype=model_dtype,
    )
    vae = components["vae"]
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    scheduler = components["scheduler"]

    # Pre-encode negative (empty) prompt
    print("Encoding negative prompt...")
    neg_embeds = encode_prompt(
        tokenizer, text_encoder, "",
        device=device, max_sequence_length=args.max_sequence_length,
        dtype=model_dtype,
    )

    # Load classifier (optional)
    classifier = None
    use_guidance = (
        args.classifier_ckpt is not None and args.guidance_scale > 0
    )
    if use_guidance:
        print(f"Loading classifier from {args.classifier_ckpt}...")
        classifier = LatentResNet18Classifier(
            num_classes=args.num_classes, pretrained_backbone=False,
        ).to(device)
        ckpt = torch.load(args.classifier_ckpt, map_location=device)
        classifier.load_state_dict(ckpt)
        classifier.eval()
        print(f"Classifier loaded ({args.num_classes} classes)")

    # Read prompts
    if args.prompt_csv:
        import csv
        with open(args.prompt_csv) as f:
            reader = csv.DictReader(f)
            prompts = [row[args.csv_column].strip() for row in reader
                       if row[args.csv_column].strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_csv} (column: '{args.csv_column}')")
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        raise ValueError("Either --prompt_file or --prompt_csv is required")

    os.makedirs(args.output_dir, exist_ok=True)

    mode_str = "guided" if use_guidance else "baseline"
    print(f"\nMode: {mode_str}")
    if use_guidance:
        print(f"  guidance_scale={args.guidance_scale}")
        print(f"  guidance_mode={args.guidance_mode}")
        print(f"  guidance_start_step={args.guidance_start_step}")

    # Save prompts for reference
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        for p in prompts:
            f.write(p + "\n")

    for idx, prompt in enumerate(prompts):
        print(f"\n=== [{idx+1}/{len(prompts)}] \"{prompt}\" ===")

        prompt_embeds = encode_prompt(
            tokenizer, text_encoder, prompt,
            device=device, max_sequence_length=args.max_sequence_length,
            dtype=model_dtype,
        )

        for s in range(args.nsamples):
            seed = args.seed + idx * 1000 + s
            generator = torch.Generator(device=device).manual_seed(seed)

            with torch.enable_grad():
                images = guided_euler_sample(
                    transformer=transformer,
                    vae=vae,
                    scheduler=scheduler,
                    classifier=classifier,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    num_steps=args.num_inference_steps,
                    cfg_scale=args.cfg_scale,
                    guidance_scale=args.guidance_scale if use_guidance else 0.0,
                    guidance_start_step=args.guidance_start_step,
                    target_class=args.target_class,
                    guidance_mode=args.guidance_mode,
                    safe_classes=args.safe_classes,
                    harm_classes=args.harm_classes,
                    grad_clip_ratio=args.grad_clip_ratio,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    device=device,
                    model_dtype=model_dtype,
                    verbose=(s == 0),
                )

            img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            fname = f"{idx:06d}.png" if args.nsamples == 1 else f"{idx:06d}_{s}.png"
            Image.fromarray(img).save(os.path.join(args.output_dir, fname))

    print(f"\nDone. {len(prompts)} prompts x {args.nsamples} samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
