#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate images with z0 classifier guidance for REPA/SiT.

Injects classifier guidance into the Euler ODE/SDE sampler:
  1. CFG velocity prediction (no_grad)
  2. Predict x0_hat = x_t - t * v_pred (detached)
  3. Classifier(x0_hat) -> log p(target_class)
  4. Gradient d(log_prob)/d(x_t) = d(log_prob)/d(x0_hat)  [identity Jacobian!]
  5. v_guided = v + guidance_scale * grad
  6. Euler step: x_{t-dt} = x_t + dt * v_guided

Key advantage over SD: d(x0_hat)/d(x_t) = I (identity matrix).
No 1/sqrt(alpha_bar) amplification -> stable gradients at all timesteps.
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# REPA model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repa_src"))
from models.sit import SiT_models
from utils import download_model
from diffusers.models import AutoencoderKL

# REPA samplers
from samplers import (
    get_score_from_velocity,
    compute_diffusion,
    euler_maruyama_sampler,
    euler_sampler,
)

# Local
from classifiers.latent_classifier import LatentResNet18Classifier

VAE_SCALE = 0.18215
NULL_CLASS = 1000


def guided_euler_sampler(
    model,
    latents,
    y,
    classifier,
    num_steps=250,
    cfg_scale=1.8,
    guidance_scale=5.0,
    guidance_low=0.0,
    guidance_high=1.0,
    clf_guidance_start=0.0,
    clf_guidance_end=1.0,
    target_class=1,
    guidance_mode="target",
    safe_classes=None,
    harm_classes=None,
    grad_clip_ratio=0.3,
    verbose=True,
):
    """
    Euler ODE sampler with classifier guidance for REPA/SiT.

    Args:
        model: SiT model
        latents: (B, 4, 32, 32) initial noise
        y: (B,) ImageNet class labels
        classifier: trained z0 classifier
        num_steps: number of Euler steps
        cfg_scale: classifier-free guidance scale
        guidance_scale: classifier guidance strength
        guidance_low/high: CFG time window
        clf_guidance_start/end: classifier guidance time window (t values)
        target_class: class to guide toward (0=benign, 1=clothed, 2=nude)
        guidance_mode: "target" or "safe_minus_harm"
        safe_classes: list of safe class indices (for safe_minus_harm mode)
        harm_classes: list of harm class indices (for safe_minus_harm mode)
        grad_clip_ratio: max gradient norm as fraction of velocity norm
        verbose: print guidance info
    """
    device = latents.device
    _dtype = latents.dtype
    y_null = torch.tensor([NULL_CLASS] * y.size(0), device=device)

    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.detach()

        # ---- CFG velocity prediction (no_grad) ----
        with torch.no_grad():
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y

            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
            v_pred_all = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur
            )[0].to(torch.float64)

            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                v_cond, v_uncond = v_pred_all.chunk(2)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_pred_all

        # ---- Classifier guidance ----
        should_guide = (
            guidance_scale > 0
            and t_cur >= clf_guidance_start
            and t_cur <= clf_guidance_end
        )

        if should_guide:
            x_cur_grad = x_cur.clone().detach().float().requires_grad_(True)
            v_det = v_pred.detach().float()

            # x0_hat = x_t - t * v_pred  (d(x0)/d(x_t) = I)
            x0_hat = x_cur_grad - float(t_cur) * v_det
            logits = classifier(x0_hat)
            log_probs = F.log_softmax(logits, dim=-1)

            if guidance_mode == "safe_minus_harm":
                safe_cls = safe_classes or [0, 1]
                harm_cls = harm_classes or [2]
                safe_lse = torch.logsumexp(log_probs[:, safe_cls], dim=-1)
                harm_lse = torch.logsumexp(log_probs[:, harm_cls], dim=-1)
                diff_val = (safe_lse - harm_lse).sum()
            else:
                diff_val = log_probs[:, target_class].sum()

            grad = torch.autograd.grad(diff_val, x_cur_grad)[0]
            grad = grad.to(torch.float64)

            # Gradient clipping
            grad_norm = grad.norm()
            v_norm = v_pred.norm()
            max_grad = v_norm * grad_clip_ratio
            clipped = False
            if grad_norm > max_grad and grad_norm > 0:
                grad = grad * (max_grad / grad_norm)
                clipped = True

            if verbose and i % 50 == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    prob_str = " ".join([f"c{c}={probs[0,c]:.3f}" for c in range(logits.shape[1])])
                print(
                    f"  step={i:3d}, t={float(t_cur):.4f}, ||grad||={grad_norm:.4f}, "
                    f"||v||={v_norm:.4f}, clipped={clipped}, {prob_str}"
                )

            # Apply guidance to velocity
            v_pred = v_pred + guidance_scale * grad

        # ---- Euler step ----
        x_next = x_cur + (t_next - t_cur) * v_pred

    return x_next


def guided_euler_maruyama_sampler(
    model,
    latents,
    y,
    classifier,
    num_steps=250,
    cfg_scale=1.8,
    guidance_scale=5.0,
    guidance_low=0.0,
    guidance_high=1.0,
    clf_guidance_start=0.0,
    clf_guidance_end=1.0,
    target_class=1,
    guidance_mode="target",
    safe_classes=None,
    harm_classes=None,
    grad_clip_ratio=0.3,
    path_type="linear",
    verbose=True,
):
    """
    Euler-Maruyama SDE sampler with classifier guidance for REPA/SiT.

    Follows REPA's official evaluation setting (SDE mode).
    t: 1.0 -> 0.04 (SDE with noise), then 0.04 -> 0.0 (deterministic last step).
    """
    device = latents.device
    _dtype = latents.dtype
    y_null = torch.tensor([NULL_CLASS] * y.size(0), device=device)

    # Time schedule: 1.0 -> 0.04 (SDE), then append 0.0 (final ODE step)
    t_steps = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.0], dtype=torch.float64)])
    x_next = latents.to(torch.float64)

    step_idx = 0

    # ============================================================
    # SDE steps: t = 1.0 -> 0.04 (with stochastic noise)
    # ============================================================
    for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
        dt = t_next - t_cur
        x_cur = x_next.detach()
        diffusion = compute_diffusion(t_cur)
        eps_i = torch.randn_like(x_cur).to(device)
        deps = eps_i * torch.sqrt(torch.abs(dt))

        # ---- Model forward + score + drift (no_grad) ----
        with torch.no_grad():
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y

            time_input = (
                torch.ones(model_input.size(0), device=device, dtype=torch.float64)
                * t_cur
            )

            # Velocity prediction
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur
            )[0].to(torch.float64)

            # Velocity -> Score
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)

            # Drift = v - 0.5 * g^2 * score
            d_cur = v_cur - 0.5 * diffusion * s_cur

            # CFG on drift
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
                # Also get CFG velocity for x0_hat computation
                v_cond, v_uncond = v_cur.chunk(2)
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_cfg = v_cur

        # ---- Classifier guidance ----
        should_guide = (
            guidance_scale > 0
            and t_cur >= clf_guidance_start
            and t_cur <= clf_guidance_end
        )

        if should_guide:
            x_cur_grad = x_cur.clone().detach().float().requires_grad_(True)
            v_det = v_cfg.detach().float()

            # x0_hat = x_t - t * v_pred  (identity Jacobian)
            x0_hat = x_cur_grad - float(t_cur) * v_det
            logits = classifier(x0_hat)
            log_probs = F.log_softmax(logits, dim=-1)

            if guidance_mode == "safe_minus_harm":
                safe_cls = safe_classes or [0, 1]
                harm_cls = harm_classes or [2]
                safe_lse = torch.logsumexp(log_probs[:, safe_cls], dim=-1)
                harm_lse = torch.logsumexp(log_probs[:, harm_cls], dim=-1)
                diff_val = (safe_lse - harm_lse).sum()
            else:
                diff_val = log_probs[:, target_class].sum()

            grad = torch.autograd.grad(diff_val, x_cur_grad)[0]
            grad = grad.to(torch.float64)

            # Gradient clipping
            grad_norm = grad.norm()
            d_norm = d_cur.norm()
            max_grad = d_norm * grad_clip_ratio
            clipped = False
            if grad_norm > max_grad and grad_norm > 0:
                grad = grad * (max_grad / grad_norm)
                clipped = True

            if verbose and step_idx % 50 == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    prob_str = " ".join(
                        [f"c{c}={probs[0,c]:.3f}" for c in range(logits.shape[1])]
                    )
                print(
                    f"  step={step_idx:3d}, t={float(t_cur):.4f}, ||grad||={grad_norm:.4f}, "
                    f"||d||={d_norm:.4f}, clipped={clipped}, {prob_str}"
                )

            # Apply guidance to drift
            d_cur = d_cur + guidance_scale * grad

        # ---- SDE step ----
        x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
        step_idx += 1

    # ============================================================
    # Last step: 0.04 -> 0.0 (deterministic, no noise)
    # ============================================================
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next.detach()

    with torch.no_grad():
        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            model_input = torch.cat([x_cur] * 2, dim=0)
            y_cur = torch.cat([y, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = y

        time_input = (
            torch.ones(model_input.size(0), device=device, dtype=torch.float64)
            * t_cur
        )

        v_cur = model(
            model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur
        )[0].to(torch.float64)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        diffusion = compute_diffusion(t_cur)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur_cond, d_cur_uncond = d_cur.chunk(2)
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            v_cond, v_uncond = v_cur.chunk(2)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_cfg = v_cur

    # Classifier guidance on last step
    should_guide = (
        guidance_scale > 0
        and t_cur >= clf_guidance_start
        and t_cur <= clf_guidance_end
    )

    if should_guide:
        x_cur_grad = x_cur.clone().detach().float().requires_grad_(True)
        v_det = v_cfg.detach().float()
        x0_hat = x_cur_grad - float(t_cur) * v_det
        logits = classifier(x0_hat)
        log_probs = F.log_softmax(logits, dim=-1)

        if guidance_mode == "safe_minus_harm":
            safe_cls = safe_classes or [0, 1]
            harm_cls = harm_classes or [2]
            safe_lse = torch.logsumexp(log_probs[:, safe_cls], dim=-1)
            harm_lse = torch.logsumexp(log_probs[:, harm_cls], dim=-1)
            diff_val = (safe_lse - harm_lse).sum()
        else:
            diff_val = log_probs[:, target_class].sum()

        grad = torch.autograd.grad(diff_val, x_cur_grad)[0].to(torch.float64)

        grad_norm = grad.norm()
        d_norm = d_cur.norm()
        max_grad = d_norm * grad_clip_ratio
        if grad_norm > max_grad and grad_norm > 0:
            grad = grad * (max_grad / grad_norm)

        d_cur = d_cur + guidance_scale * grad

    # Deterministic step (mean_x, no noise)
    x_next = x_cur + dt * d_cur

    return x_next


# Person-related ImageNet classes for testing
PERSON_CLASSES = {
    445: "bikini",
    459: "brassiere",
    614: "kimono",
    638: "maillot_1",
    639: "maillot_2",
    677: "miniskirt",
    834: "suit",
    842: "swimming_trunks",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image generation with z0 classifier guidance for REPA/SiT"
    )
    parser.add_argument("--output_dir", type=str, default="output_img/guided")
    parser.add_argument("--nsamples", type=int, default=4,
                        help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42)
    # SiT model
    parser.add_argument("--sit_model", type=str, default="SiT-XL/2")
    parser.add_argument("--sit_ckpt", type=str, default=None)
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae_type", type=str, default="ema", choices=["ema", "mse"])
    # Sampling
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=1.8)
    parser.add_argument("--guidance_low", type=float, default=0.0)
    parser.add_argument("--guidance_high", type=float, default=0.7,
                        help="CFG guidance interval (original REPA default)")
    # Classifier guidance
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Path to trained z0 classifier. None = no guidance")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--clf_guidance_start", type=float, default=0.0,
                        help="Start classifier guidance at this t value")
    parser.add_argument("--clf_guidance_end", type=float, default=1.0,
                        help="End classifier guidance at this t value")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class (0=benign, 1=clothed, 2=nude)")
    parser.add_argument("--guidance_mode", type=str, default="target",
                        choices=["target", "safe_minus_harm"])
    parser.add_argument("--safe_classes", type=int, nargs="+", default=None)
    parser.add_argument("--harm_classes", type=int, nargs="+", default=None)
    parser.add_argument("--grad_clip_ratio", type=float, default=0.3)
    # Sampler mode
    parser.add_argument("--sampler_mode", type=str, default="sde",
                        choices=["ode", "sde"],
                        help="Sampling mode: 'sde' (Euler-Maruyama, recommended) or 'ode' (Euler)")
    # Which ImageNet classes to generate
    parser.add_argument("--imagenet_classes", type=int, nargs="+", default=None,
                        help="ImageNet class indices. None = all person-related classes")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    latent_size = args.resolution // 8  # 32

    # Load SiT model
    print("Loading SiT model...")
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    sit_model = SiT_models[args.sit_model](
        input_size=latent_size,
        num_classes=1000,
        use_cfg=True,
        z_dims=[768],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    if args.sit_ckpt is None:
        state_dict = download_model("last.pt")
    else:
        state_dict = torch.load(args.sit_ckpt, map_location=device)
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
    sit_model.load_state_dict(state_dict)
    sit_model.eval()
    print(f"SiT loaded: {sum(p.numel() for p in sit_model.parameters()):,} params")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae_type}"
    ).to(device)

    # Load classifier (optional)
    classifier = None
    use_guidance = args.classifier_ckpt is not None and args.guidance_scale > 0
    if use_guidance:
        print(f"Loading classifier from {args.classifier_ckpt}...")
        classifier = LatentResNet18Classifier(
            num_classes=args.num_classes, pretrained_backbone=False
        ).to(device)
        ckpt = torch.load(args.classifier_ckpt, map_location=device)
        classifier.load_state_dict(ckpt)
        classifier.eval()
        print(f"Classifier loaded ({args.num_classes} classes)")

    # VAE decode helpers
    latents_scale = torch.tensor([VAE_SCALE] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.zeros(1, 4, 1, 1).to(device)

    # Determine classes to generate
    classes_to_gen = args.imagenet_classes or list(PERSON_CLASSES.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    mode_str = "guided" if use_guidance else "baseline"
    print(f"\nMode: {mode_str}, Sampler: {args.sampler_mode.upper()}")
    if use_guidance:
        print(f"  guidance_scale={args.guidance_scale}, target_class={args.target_class}")
        print(f"  guidance_mode={args.guidance_mode}")
        print(f"  clf_guidance_range=[{args.clf_guidance_start}, {args.clf_guidance_end}]")

    for cls_idx in classes_to_gen:
        cls_name = PERSON_CLASSES.get(cls_idx, f"class_{cls_idx}")
        print(f"\n=== Class {cls_idx}: {cls_name} ===")

        for sample_idx in range(args.nsamples):
            # Deterministic seed per sample for reproducibility
            torch.manual_seed(args.seed + cls_idx * 1000 + sample_idx)
            z = torch.randn(1, 4, latent_size, latent_size, device=device)
            y = torch.tensor([cls_idx], device=device)

            if use_guidance:
                # Select guided sampler based on mode
                guided_fn = (
                    guided_euler_maruyama_sampler
                    if args.sampler_mode == "sde"
                    else guided_euler_sampler
                )
                samples = guided_fn(
                    model=sit_model,
                    latents=z,
                    y=y,
                    classifier=classifier,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    guidance_scale=args.guidance_scale,
                    guidance_low=args.guidance_low,
                    guidance_high=args.guidance_high,
                    clf_guidance_start=args.clf_guidance_start,
                    clf_guidance_end=args.clf_guidance_end,
                    target_class=args.target_class,
                    guidance_mode=args.guidance_mode,
                    safe_classes=args.safe_classes,
                    harm_classes=args.harm_classes,
                    grad_clip_ratio=args.grad_clip_ratio,
                    verbose=(sample_idx == 0),
                ).to(torch.float32)
            else:
                # Baseline: use original REPA sampler (no guidance)
                baseline_fn = (
                    euler_maruyama_sampler
                    if args.sampler_mode == "sde"
                    else euler_sampler
                )
                with torch.no_grad():
                    samples = baseline_fn(
                        model=sit_model,
                        latents=z,
                        y=y,
                        num_steps=args.num_steps,
                        cfg_scale=args.cfg_scale,
                        guidance_low=args.guidance_low,
                        guidance_high=args.guidance_high,
                    ).to(torch.float32)

            # Decode
            with torch.no_grad():
                images = vae.decode((samples - latents_bias) / latents_scale).sample
                images = (images + 1) / 2.0
                images = images.clamp(0, 1)
                img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                fname = f"{cls_name}_{cls_idx}_sample{sample_idx}.png"
                Image.fromarray(img).save(os.path.join(args.output_dir, fname))

        print(f"  {args.nsamples} samples saved")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
