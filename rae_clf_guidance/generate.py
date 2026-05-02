#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate images with z0 classifier guidance for RAE/DiTDH.

Injects classifier guidance into a custom Euler ODE/SDE sampler:
  1. CFG velocity prediction (no_grad) via DiTDH
  2. Predict x0_hat = x_t - t * v_pred (detached)
  3. Classifier(x0_hat) -> log p(target_class)
  4. Gradient d(log_prob)/d(x_t) = d(log_prob)/d(x0_hat)  [identity Jacobian!]
  5. v_guided = v + guidance_scale * grad
  6. Euler step: x_{t-dt} = x_t + dt * v_guided

Key differences from REPA version:
  - DiTDH returns velocity tensor directly (NOT tuple like SiT)
  - Shifted time schedule: t' = shift*t/(1+(shift-1)*t), shift≈6.928
  - RAE decode (DINOv2 latent -> image) instead of VAE decode
  - Latent shape: [768, 16, 16] instead of [4, 32, 32]
"""

import os
import sys
import math
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# RAE model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rae_src"))
from stage1 import RAE
from stage2.models.DDT import DiTwDDTHead

# Local
from classifiers.latent_classifier import DINOv2LatentClassifier

NULL_CLASS = 1000
LATENT_SHAPE = (768, 16, 16)


def get_score_from_velocity(v, x, t, path_type="linear"):
    """Convert velocity to score for linear flow matching path.
    score(x_t, t) = -(v*(1-t) + x_t) / t
    """
    if path_type == "linear":
        return -(v * (1 - t) + x) / t
    raise NotImplementedError(f"Unknown path_type: {path_type}")


def compute_diffusion(t):
    """Diffusion coefficient g^2(t) for SBDM matching."""
    return 2 * t * (1 - t)


def make_shifted_time_schedule(num_steps, time_dist_shift, t0=0.0, t1=0.999):
    """Create shifted time schedule for RAE sampling.

    Returns tensor of length num_steps+1 going from ~1.0 to ~0.0,
    with non-uniform spacing from the time distribution shift.
    """
    t_linear = 1.0 - torch.linspace(t0, t1, num_steps + 1, dtype=torch.float64)
    t_shifted = time_dist_shift * t_linear / (1 + (time_dist_shift - 1) * t_linear)
    return t_shifted


def guided_euler_sampler(
    model,
    latents,
    y,
    classifier,
    num_steps=50,
    cfg_scale=1.75,
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
    time_dist_shift=6.928,
    verbose=True,
):
    """
    Euler ODE sampler with classifier guidance for RAE/DiTDH.

    Uses shifted time schedule matching RAE's training distribution.
    DiTDH forward returns velocity tensor directly (NOT tuple).
    """
    device = latents.device
    _dtype = latents.dtype
    y_null = torch.tensor([NULL_CLASS] * y.size(0), device=device)

    t_steps = make_shifted_time_schedule(num_steps, time_dist_shift)
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
            # DiTDH returns velocity tensor directly (NOT tuple like SiT!)
            v_pred_all = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur
            ).to(torch.float64)

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

            if verbose and i % 10 == 0:
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
    num_steps=50,
    cfg_scale=1.75,
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
    time_dist_shift=6.928,
    path_type="linear",
    verbose=True,
):
    """
    Euler-Maruyama SDE sampler with classifier guidance for RAE/DiTDH.

    Time steps go from ~1 (noisy) to ~0 (clean) with shifted schedule.
    SDE portion uses noise, final step is deterministic.
    """
    device = latents.device
    _dtype = latents.dtype
    y_null = torch.tensor([NULL_CLASS] * y.size(0), device=device)

    # Shifted time schedule: main SDE steps + final deterministic step
    # Use num_steps-1 SDE steps, then one final ODE step to 0
    t_steps = make_shifted_time_schedule(num_steps, time_dist_shift)
    x_next = latents.to(torch.float64)

    step_idx = 0

    # ============================================================
    # SDE steps (with stochastic noise) — all but last step
    # ============================================================
    for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
        dt = t_next - t_cur  # negative (t decreasing)
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

            # DiTDH returns velocity tensor directly (NOT tuple!)
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), y=y_cur
            ).to(torch.float64)

            # Velocity -> Score
            s_cur = get_score_from_velocity(v_cur, model_input.to(torch.float64), time_input, path_type=path_type)

            # Drift = v - 0.5 * g^2 * score
            d_cur = v_cur - 0.5 * diffusion * s_cur

            # CFG on drift
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
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

            grad_norm = grad.norm()
            d_norm = d_cur.norm()
            max_grad = d_norm * grad_clip_ratio
            clipped = False
            if grad_norm > max_grad and grad_norm > 0:
                grad = grad * (max_grad / grad_norm)
                clipped = True

            if verbose and step_idx % 10 == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    prob_str = " ".join(
                        [f"c{c}={probs[0,c]:.3f}" for c in range(logits.shape[1])]
                    )
                print(
                    f"  step={step_idx:3d}, t={float(t_cur):.4f}, ||grad||={grad_norm:.4f}, "
                    f"||d||={d_norm:.4f}, clipped={clipped}, {prob_str}"
                )

            d_cur = d_cur + guidance_scale * grad

        # ---- SDE step ----
        x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
        step_idx += 1

    # ============================================================
    # Last step: deterministic (no noise)
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
        ).to(torch.float64)

        s_cur = get_score_from_velocity(v_cur, model_input.to(torch.float64), time_input, path_type=path_type)
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

    # Deterministic step
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
        description="Image generation with z0 classifier guidance for RAE/DiTDH"
    )
    parser.add_argument("--output_dir", type=str, default="output_img/guided")
    parser.add_argument("--nsamples", type=int, default=4,
                        help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42)
    # RAE model
    parser.add_argument("--decoder_config_path", type=str,
                        default="rae_src/configs/decoder/ViTXL")
    parser.add_argument("--decoder_ckpt", type=str, required=True,
                        help="Path to RAE decoder checkpoint")
    parser.add_argument("--stat_path", type=str, required=True,
                        help="Path to DINOv2 latent normalization stats (stat.pt)")
    # DiTDH model
    parser.add_argument("--ditdh_ckpt", type=str, required=True,
                        help="Path to DiTDH-XL checkpoint")
    # Sampling
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.75)
    parser.add_argument("--guidance_low", type=float, default=0.0)
    parser.add_argument("--guidance_high", type=float, default=1.0,
                        help="CFG guidance interval upper bound")
    # Time distribution shift
    parser.add_argument("--time_shift_dim", type=int, default=196608,
                        help="768*16*16=196608")
    parser.add_argument("--time_shift_base", type=int, default=4096)
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
    parser.add_argument("--sampler_mode", type=str, default="ode",
                        choices=["ode", "sde"],
                        help="Sampling mode: 'ode' (Euler, default for RAE) or 'sde' (Euler-Maruyama)")
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

    time_dist_shift = math.sqrt(args.time_shift_dim / args.time_shift_base)
    print(f"time_dist_shift = {time_dist_shift:.4f}")

    # Load RAE (with decoder for image generation)
    print("Loading RAE (DINOv2 encoder + ViT decoder)...")
    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-with-registers-base',
        encoder_input_size=224,
        encoder_params={
            'dinov2_path': 'facebook/dinov2-with-registers-base',
            'normalize': True,
        },
        decoder_config_path=args.decoder_config_path,
        pretrained_decoder_path=args.decoder_ckpt,
        noise_tau=0.,
        reshape_to_2d=True,
        normalization_stat_path=args.stat_path,
    )
    rae.eval()
    rae.to(device)
    print("RAE loaded")

    # Load DiTDH-XL model
    print("Loading DiTDH-XL...")
    ditdh = DiTwDDTHead(
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    ).to(device)

    state_dict = torch.load(args.ditdh_ckpt, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    ditdh.load_state_dict(state_dict, strict=True)
    ditdh.eval()
    print(f"DiTDH-XL loaded: {sum(p.numel() for p in ditdh.parameters()):,} params")

    # Load classifier (optional)
    classifier = None
    use_guidance = args.classifier_ckpt is not None and args.guidance_scale > 0
    if use_guidance:
        print(f"Loading classifier from {args.classifier_ckpt}...")
        classifier = DINOv2LatentClassifier(
            in_channels=768,
            num_classes=args.num_classes,
        ).to(device)
        ckpt = torch.load(args.classifier_ckpt, map_location=device)
        classifier.load_state_dict(ckpt)
        classifier.eval()
        print(f"Classifier loaded ({args.num_classes} classes)")

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
            torch.manual_seed(args.seed + cls_idx * 1000 + sample_idx)
            z = torch.randn(1, *LATENT_SHAPE, device=device)
            y = torch.tensor([cls_idx], device=device)

            if use_guidance:
                guided_fn = (
                    guided_euler_maruyama_sampler
                    if args.sampler_mode == "sde"
                    else guided_euler_sampler
                )
                samples = guided_fn(
                    model=ditdh,
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
                    time_dist_shift=time_dist_shift,
                    verbose=(sample_idx == 0),
                ).to(torch.float32)
            else:
                # Baseline: Euler ODE without classifier guidance
                samples = guided_euler_sampler(
                    model=ditdh,
                    latents=z,
                    y=y,
                    classifier=None,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    guidance_scale=0.0,  # no classifier guidance
                    guidance_low=args.guidance_low,
                    guidance_high=args.guidance_high,
                    time_dist_shift=time_dist_shift,
                    verbose=False,
                ).to(torch.float32)

            # Decode with RAE
            with torch.no_grad():
                images = rae.decode(samples)
                images = images.clamp(0, 1)
                img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                fname = f"{cls_name}_{cls_idx}_sample{sample_idx}.png"
                Image.fromarray(img).save(os.path.join(args.output_dir, fname))

        print(f"  {args.nsamples} samples saved")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
