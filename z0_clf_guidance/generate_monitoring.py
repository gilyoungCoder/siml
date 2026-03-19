#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Z0 ASCG: Adaptive Spatial Classifier Guidance with Monitoring.

Implements the ASCG algorithm adapted for z0 classifier guidance:

At each denoising step:
  1. Tweedie: zt → z0_hat (clean latent prediction)
  2. GradCAM on z0_hat → saliency map s_{t,c}
  3. Monitoring: P̄(harm) = Φ((mean(s) - μ_s̄) / σ_s̄)
     - If P̄(harm) ≤ τ̄: skip guidance (preserve image quality)
  4. Spatial mask: m_t = 1[CDF_pixel(s) ≥ τ_spatial]
  5. Gradient: g = ∇_zt [log p(safe | z0_hat) - log p(harm | z0_hat)]
  6. Spatial weighting: g_spatial = g * (m * λ_strong + (1-m) * λ_weak)
  7. Score-based guidance: ε̂ = ε - √(1-ᾱ) · g_spatial

Monitoring modes:
  - "classifier": P(harm) = softmax(logits)[harm_class] (no pre-computed stats needed)
  - "gradcam": P(harm) = CDF((mean(GradCAM) - μ) / σ) (needs harmful_stats.pt)

Usage:
    python generate_monitoring.py \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file /path/to/prompts.csv \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --harmful_stats_path ./harmful_stats.pt \
        --monitoring_threshold 0.3 \
        --guidance_scale 15.0 \
        --base_guidance_scale 1.0 \
        --spatial_threshold_start 0.3 \
        --spatial_threshold_end 0.3 \
        --output_dir ./output
"""

import csv
import json
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.attention_utils import (
    AttentionStore, register_attention_store,
    compute_attention_mask, detect_harmful_tokens,
)
from models.latent_classifier import LatentResNet18Classifier
from utils.denoise_utils import predict_z0


def parse_args():
    parser = ArgumentParser(description="Z0 ASCG with monitoring")
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_monitoring")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--gradcam_layer", type=str, default="layer2",
                        choices=["layer1", "layer2", "layer3", "layer4"])

    # Monitoring (selective intervention)
    parser.add_argument("--monitoring_mode", type=str, default="classifier",
                        choices=["classifier", "gradcam", "dual"],
                        help="'classifier': softmax P(harm). 'gradcam': CDF of GradCAM mean. 'dual': softmax sticky + GradCAM CDF per-step.")
    parser.add_argument("--monitoring_threshold", type=float, default=0.3,
                        help="Threshold for triggering guidance (0-1). In dual mode: softmax sticky trigger threshold.")
    parser.add_argument("--cdf_threshold", type=float, default=0.05,
                        help="GradCAM CDF per-step threshold (only used in dual mode)")

    # Guidance
    parser.add_argument("--guidance_scale", type=float, default=15.0,
                        help="Guidance strength inside harmful spatial regions (λ_strong)")
    parser.add_argument("--base_guidance_scale", type=float, default=1.0,
                        help="Guidance strength outside harmful regions (λ_weak)")
    parser.add_argument("--harm_ratio", type=float, default=1.0,
                        help="λ_rep: weight for harm repulsion in dual gradient (Eq.6: g = ∇log p(safe) - λ_rep·∇log p(harm))")

    # Spatial masking
    parser.add_argument("--harmful_stats_path", type=str, default=None,
                        help="Path to harmful_stats.pt for CDF spatial thresholding")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.3,
                        help="Spatial CDF threshold at start of denoising")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="Spatial CDF threshold at end of denoising")
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine",
                        choices=["constant", "linear", "cosine"])

    # Step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    # Gradient clipping
    parser.add_argument("--grad_clip_ratio", type=float, default=0.3,
                        help="Max guidance norm as fraction of noise_pred norm (0=disable)")

    # Trigger modes
    parser.add_argument("--sticky_trigger", action="store_true",
                        help="Once P(harm) > threshold at any step, guide ALL remaining steps")
    parser.add_argument("--always_guide", action="store_true",
                        help="Skip monitoring, apply guidance at every step")
    parser.add_argument("--monitoring_only", action="store_true",
                        help="Record classifier scores only, skip all guidance (for threshold analysis)")

    # Spatial mask type
    parser.add_argument("--spatial_soft", action="store_true",
                        help="Use soft (continuous CDF) spatial mask instead of binary threshold")

    # Monitoring start step
    parser.add_argument("--monitoring_start_step", type=int, default=0,
                        help="Start monitoring from this step (skip earlier steps for cleaner separation)")

    # Spatial mode
    parser.add_argument("--spatial_mode", type=str, default="gradcam",
                        choices=["gradcam", "cross_attn"],
                        help="'gradcam': classifier GradCAM mask (default). "
                             "'cross_attn': cross-attention maps of auto-detected harmful tokens.")
    parser.add_argument("--attn_resolutions", nargs="+", type=int, default=[16, 32],
                        help="UNet cross-attention resolutions to hook (for cross_attn mode)")
    parser.add_argument("--harmful_top_k", type=int, default=5,
                        help="Number of top harmful tokens to detect (for cross_attn mode)")
    parser.add_argument("--attn_sharpness", type=float, default=3.0,
                        help="Power scaling for attention mask sharpening (higher = more selective)")

    # Prompt slicing for multi-GPU
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_prompts(f):
    """Load prompts from txt or CSV file."""
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
            prompt_col = None
            for col in column_priority:
                if col in fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"CSV has no recognizable prompt column. Available: {fieldnames}")
            print(f"[INFO] Using column '{prompt_col}' from {f}")
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    else:
        return [l.strip() for l in open(f) if l.strip()]


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.resize((512, 512)).save(path)


def get_spatial_threshold(step, total, start, end, strategy="cosine"):
    """Compute spatial threshold based on denoising progress."""
    t = step / max(total - 1, 1)
    if strategy == "constant":
        return start
    elif strategy == "linear":
        return start - (start - end) * t
    elif strategy == "cosine":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


class Z0Monitor:
    """
    Monitoring module: decides WHETHER to apply guidance at each step.

    Two modes:
      - "classifier": P(harm) = softmax(classifier(z0_hat))[harm_class]
        Simple and requires no pre-computed stats.
      - "gradcam": P(harm) = CDF((mean(GradCAM(z0_hat)) - μ) / σ)
        Uses sample-level stats from harmful training images.
    """

    def __init__(self, classifier, mode="classifier", harm_class=2,
                 harm_classes=None, gradcam_layer="layer2",
                 harmful_stats=None, device="cuda"):
        self.classifier = classifier
        self.mode = mode
        self.harm_class = harm_class
        self.harm_classes = harm_classes or [harm_class]
        self.gradcam_layer = gradcam_layer
        self.device = device

        # Load monitoring stats for gradcam mode
        self.mon_mu = 0.0
        self.mon_sigma = 1.0
        if mode in ("gradcam", "dual") and harmful_stats is not None:
            # Use full-image GradCAM stats for monitoring
            # sample_level stats are better if available
            if "sample_level_mu" in harmful_stats:
                self.mon_mu = harmful_stats["sample_level_mu"]
                self.mon_sigma = harmful_stats["sample_level_sigma"]
            else:
                self.mon_mu = harmful_stats["gradcam_full_mu"]
                self.mon_sigma = harmful_stats["gradcam_full_sigma"]
            print(f"  [monitor] GradCAM CDF mode: μ={self.mon_mu:.4f}, σ={self.mon_sigma:.4f}")

        # Stats tracking
        self.stats = {"total": 0, "guided": 0, "skipped": 0, "history": []}

    def compute_p_harm(self, z0_hat, gradcam_map=None):
        """
        Compute P(harm) for monitoring.

        Returns:
            p_harm: float in [0, 1]
            gradcam_map: (B, 1, H, W) if computed, else None
        """
        if self.mode == "classifier":
            with torch.no_grad():
                logits = self.classifier(z0_hat)
                probs = F.softmax(logits, dim=-1)
                p_harm = sum(probs[:, c].mean().item() for c in self.harm_classes)
            return p_harm, None

        elif self.mode == "gradcam":
            # Compute GradCAM if not already provided
            if gradcam_map is None:
                with torch.enable_grad():
                    gradcam_map = self.classifier.compute_gradcam(
                        z0_hat.detach(), target_class=self.harm_class,
                        layer_name=self.gradcam_layer
                    )
            heatmap_mean = gradcam_map.mean().item()
            z = (heatmap_mean - self.mon_mu) / (self.mon_sigma + 1e-8)
            p_harm = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            return p_harm, gradcam_map

    def should_guide(self, z0_hat, threshold, step, gradcam_map=None):
        """Determine if guidance should be applied."""
        self.stats["total"] += 1
        p_harm, gcam = self.compute_p_harm(z0_hat, gradcam_map)
        should = p_harm > threshold
        if should:
            self.stats["guided"] += 1
        else:
            self.stats["skipped"] += 1
        self.stats["history"].append({
            "step": step, "p_harm": p_harm, "guided": should
        })
        return should, p_harm, gcam

    def reset(self):
        self.stats = {"total": 0, "guided": 0, "skipped": 0, "history": []}


class Z0SpatialGuidance:
    """
    Spatial guidance module: computes WHERE and HOW MUCH to guide.

    Uses GradCAM saliency maps for spatial masking and dual gradient
    (safe - harm) for steering direction.
    """

    def __init__(self, classifier, harmful_stats=None, gradcam_layer="layer2",
                 harm_class=2, harm_classes=None, safe_class=1, device="cuda"):
        self.classifier = classifier
        self.gradcam_layer = gradcam_layer
        self.harm_class = harm_class
        self.harm_classes = harm_classes or [harm_class]
        self.safe_class = safe_class
        self.device = device

        # Pixel-level CDF stats (top-K pixels from harmful training images)
        self.topk_mu = 0.0
        self.topk_sigma = 1.0
        if harmful_stats is not None and "gradcam_mu" in harmful_stats:
            self.topk_mu = harmful_stats["gradcam_mu"]
            self.topk_sigma = harmful_stats["gradcam_sigma"]
            print(f"  [spatial] Pixel CDF: μ={self.topk_mu:.4f}, σ={self.topk_sigma:.4f}")

    def compute_spatial_mask(self, gradcam_map, spatial_threshold, soft=False):
        """
        Compute spatial mask from GradCAM using pixel-level CDF.

        Args:
            gradcam_map: (B, 1, H, W) GradCAM heatmap, normalized [0,1] per sample
            spatial_threshold: CDF percentile threshold (ignored if soft=True)
            soft: If True, return continuous CDF values as mask weights

        Returns:
            mask: (B, 1, H, W) spatial mask (binary or continuous)
            mask_ratio: fraction of masked pixels (for hard) or mean weight (for soft)
        """
        z_pix = (gradcam_map - self.topk_mu) / (self.topk_sigma + 1e-8)
        cdf_pix = 0.5 * (1.0 + torch.erf(z_pix / math.sqrt(2.0)))
        if soft:
            mask = cdf_pix
        else:
            mask = (cdf_pix >= spatial_threshold).float()
        mask_ratio = mask.mean().item()
        return mask, mask_ratio

    def compute_gradient(self, classifier, z0_hat, prev_latents, noise_pred_uncond,
                         alpha_bar, gradcam_map, spatial_threshold,
                         guidance_scale, base_scale, harm_ratio=1.0, soft=False):
        """
        Compute spatially-weighted classifier gradient.

        Following SoftDelete+CG original: separate raw-logit gradients for safe/harm.
        g = ∇_zt y_safe(z0) - Σ_hc ∇_zt y_hc(z0)   (through Tweedie chain rule)

        Returns:
            grad_weighted: (B, 4, 64, 64) spatially weighted gradient w.r.t. zt
            mask_ratio: float, fraction of spatial mask
        """
        # 1. Spatial mask from GradCAM
        mask, mask_ratio = self.compute_spatial_mask(gradcam_map, spatial_threshold, soft=soft)

        # 2. Separate raw-logit gradients (matches SoftDelete+CG original)
        with torch.enable_grad():
            # Safe class gradient
            zt_s = prev_latents.clone().detach().requires_grad_(True)
            z0_s = predict_z0(zt_s, noise_pred_uncond.detach(), alpha_bar)
            g_safe = torch.autograd.grad(classifier(z0_s)[:, self.safe_class].sum(), zt_s)[0]

            # Harm class gradients (each class separately, then sum)
            g_harm = torch.zeros_like(g_safe)
            for hc in self.harm_classes:
                zt_h = prev_latents.clone().detach().requires_grad_(True)
                z0_h = predict_z0(zt_h, noise_pred_uncond.detach(), alpha_bar)
                g_harm = g_harm + torch.autograd.grad(classifier(z0_h)[:, hc].sum(), zt_h)[0]

            grad = g_safe - harm_ratio * g_harm

        grad = grad.detach()

        # 3. Spatial weighting: λ_strong inside mask, λ_weak outside
        weight = mask * guidance_scale + (1.0 - mask) * base_scale
        grad_weighted = grad * weight

        return grad_weighted, mask_ratio


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Z0 ASCG: ADAPTIVE SPATIAL CLASSIFIER GUIDANCE WITH MONITORING")
    print(f"{'='*60}")
    if args.always_guide:
        print(f"Monitoring: ALWAYS GUIDE (no monitoring)")
    else:
        if args.monitoring_mode == "dual":
            print(f"Monitoring: mode=dual, softmax_thr={args.monitoring_threshold}, cdf_thr={args.cdf_threshold}, sticky=softmax")
        else:
            print(f"Monitoring: mode={args.monitoring_mode}, threshold={args.monitoring_threshold}, sticky={args.sticky_trigger}")
    print(f"Guidance: scale={args.guidance_scale}, base={args.base_guidance_scale}, harm_ratio={args.harm_ratio}")
    print(f"Grad clip: {args.grad_clip_ratio} {'(disabled)' if args.grad_clip_ratio <= 0 else ''}")
    print(f"Spatial mode: {args.spatial_mode}")
    if args.spatial_mode == "cross_attn":
        print(f"  Attn resolutions: {args.attn_resolutions}, top_k: {args.harmful_top_k}")
    print(f"Spatial: threshold={args.spatial_threshold_start} -> {args.spatial_threshold_end} ({args.spatial_threshold_strategy})")
    print(f"Classifier: {args.classifier_ckpt}")
    print(f"{'='*60}\n")

    # Load harmful stats
    harmful_stats = None
    if args.harmful_stats_path and os.path.exists(args.harmful_stats_path):
        harmful_stats = torch.load(args.harmful_stats_path, map_location=device)
        print(f"Loaded harmful stats from {args.harmful_stats_path}")

    # Load classifier
    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes, pretrained_backbone=False
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()
    print(f"Loaded classifier: {args.classifier_ckpt}")

    # Determine harm/safe classes based on num_classes
    # 3-class: benign(0), safe(1), harm(2)
    # 4-class: benign(0), safe(1), harm_nude(2), harm_color(3)
    safe_class = 1
    harm_classes = [2] if args.num_classes == 3 else [2, 3]
    print(f"Classes: num={args.num_classes}, safe={safe_class}, harm={harm_classes}")

    # Initialize monitor and spatial guidance
    monitor = Z0Monitor(
        classifier, mode=args.monitoring_mode, harm_class=2,
        harm_classes=harm_classes, gradcam_layer=args.gradcam_layer,
        harmful_stats=harmful_stats, device=device,
    )
    spatial_guidance = Z0SpatialGuidance(
        classifier, harmful_stats=harmful_stats,
        gradcam_layer=args.gradcam_layer, harm_class=2,
        harm_classes=harm_classes, safe_class=safe_class,
        device=device,
    )

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load pipeline
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Cross-attention hooking for attention-aware guidance
    attention_store = None
    if args.spatial_mode == "cross_attn":
        attention_store = AttentionStore()
        register_attention_store(
            pipe.unet, attention_store,
            target_resolutions=args.attn_resolutions,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            monitor.reset()
            sticky_triggered = [False]  # mutable for closure

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                if step < args.guidance_start_step or step > args.guidance_end_step:
                    return callback_kwargs

                # Skip monitoring before monitoring_start_step
                if step < args.monitoring_start_step:
                    monitor.stats["total"] += 1
                    monitor.stats["skipped"] += 1
                    return callback_kwargs

                prev_latents = callback_kwargs["prev_latents"]
                latents = callback_kwargs["latents"]
                noise_pred = callback_kwargs["noise_pred"]
                noise_pred_uncond = callback_kwargs.get("noise_pred_uncond", noise_pred)

                # 1. Tweedie: zt → z0_hat
                alpha_bar = pipe.scheduler.alphas_cumprod.to(device)[timestep]
                alpha_bar_4d = alpha_bar.view(-1, 1, 1, 1)
                z0_hat = predict_z0(prev_latents, noise_pred_uncond.detach(), alpha_bar_4d)

                # 2. GradCAM (used for both monitoring and spatial masking)
                with torch.enable_grad():
                    gradcam_map = classifier.compute_gradcam(
                        z0_hat.detach(), target_class=2,
                        layer_name=args.gradcam_layer
                    )  # (B, 1, H, W)

                # 3. Monitoring: decide whether to guide
                if args.monitoring_only:
                    # Record score only, skip all guidance
                    monitor.stats["total"] += 1
                    with torch.no_grad():
                        logits = classifier(z0_hat)
                        probs = F.softmax(logits, dim=-1)
                        p_harm = sum(probs[:, c].mean().item() for c in harm_classes)
                    monitor.stats["history"].append({
                        "step": step, "p_harm": p_harm, "guided": False
                    })
                    monitor.stats["skipped"] += 1
                    if args.debug:
                        gcam_mean = gradcam_map.mean().item()
                        print(f"  step={step:2d} t={timestep.item():4d} P={p_harm:.3f} gcam={gcam_mean:.4f} -> MONITOR_ONLY")
                    return callback_kwargs

                if args.always_guide:
                    should_guide = True
                    p_harm = -1.0
                    monitor.stats["total"] += 1
                    monitor.stats["guided"] += 1
                elif args.monitoring_mode == "dual":
                    # --- Dual monitoring: softmax sticky + GradCAM CDF per-step ---
                    # Phase 1: Classifier softmax → sticky trigger
                    with torch.no_grad():
                        logits = classifier(z0_hat)
                        probs = F.softmax(logits, dim=-1)
                        p_harm_softmax = sum(probs[:, c].mean().item() for c in harm_classes)

                    if p_harm_softmax > args.monitoring_threshold:
                        sticky_triggered[0] = True

                    if not sticky_triggered[0]:
                        monitor.stats["total"] += 1
                        monitor.stats["skipped"] += 1
                        monitor.stats["history"].append({
                            "step": step, "p_harm_softmax": p_harm_softmax,
                            "p_harm_cdf": -1.0, "guided": False
                        })
                        if args.debug:
                            gcam_mean = gradcam_map.mean().item()
                            print(f"  step={step:2d} t={timestep.item():4d} P_sm={p_harm_softmax:.3f} gcam={gcam_mean:.4f} -> skip (no trigger)")
                        return callback_kwargs

                    # Phase 2: GradCAM CDF → per-step decision
                    gcam_mean = gradcam_map.mean().item()
                    z = (gcam_mean - monitor.mon_mu) / (monitor.mon_sigma + 1e-8)
                    p_harm_cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    should_guide = p_harm_cdf > args.cdf_threshold
                    p_harm = p_harm_cdf

                    monitor.stats["total"] += 1
                    if should_guide:
                        monitor.stats["guided"] += 1
                    else:
                        monitor.stats["skipped"] += 1
                    monitor.stats["history"].append({
                        "step": step, "p_harm_softmax": p_harm_softmax,
                        "p_harm_cdf": p_harm_cdf, "guided": should_guide
                    })
                else:
                    should_guide, p_harm, _ = monitor.should_guide(
                        z0_hat, args.monitoring_threshold, step, gradcam_map
                    )

                    # Sticky trigger: once triggered, guide all remaining steps
                    if should_guide:
                        sticky_triggered[0] = True
                    elif args.sticky_trigger and sticky_triggered[0]:
                        should_guide = True
                        monitor.stats["guided"] += 1
                        monitor.stats["skipped"] -= 1

                if args.debug:
                    sticky_str = " [STICKY]" if sticky_triggered[0] else ""
                    gcam_mean = gradcam_map.mean().item()
                    if args.monitoring_mode == "dual":
                        print(f"  step={step:2d} t={timestep.item():4d} P_sm={p_harm_softmax:.3f} P_cdf={p_harm_cdf:.3f} gcam={gcam_mean:.4f} -> {'GUIDE' if should_guide else 'skip'}{sticky_str}")
                    else:
                        print(f"  step={step:2d} t={timestep.item():4d} P={p_harm:.3f} gcam={gcam_mean:.4f} -> {'GUIDE' if should_guide else 'skip'}{sticky_str}")

                if not should_guide:
                    return callback_kwargs

                # 4. Spatial threshold (cosine annealing)
                spatial_thr = get_spatial_threshold(
                    step, args.num_inference_steps,
                    args.spatial_threshold_start, args.spatial_threshold_end,
                    args.spatial_threshold_strategy
                )

                # 5. Compute spatially-weighted gradient
                if args.spatial_mode == "cross_attn" and attention_store is not None:
                    # --- Cross-attention aware guidance ---
                    # 5a. Auto-detect harmful tokens via attention × GradCAM correlation
                    harmful_indices, token_scores = detect_harmful_tokens(
                        classifier, z0_hat, attention_store,
                        gradcam_layer=args.gradcam_layer,
                        top_k=args.harmful_top_k,
                    )

                    # 5b. Harmful tokens' attention map → spatial mask
                    attn_weight = compute_attention_mask(
                        attention_store, harmful_indices,
                        target_resolution=z0_hat.shape[-1],
                        threshold=spatial_thr,
                        soft=args.spatial_soft,
                    )
                    if attn_weight is None:
                        attn_weight = torch.ones(1, 1, z0_hat.shape[-2], z0_hat.shape[-1], device=device)
                    attn_weight = attn_weight.to(device)

                    # 5c. Separate raw-logit gradients (matches SoftDelete+CG original)
                    with torch.enable_grad():
                        zt_s = prev_latents.clone().detach().requires_grad_(True)
                        z0_s = predict_z0(zt_s, noise_pred_uncond.detach(), alpha_bar_4d)
                        g_safe = torch.autograd.grad(classifier(z0_s)[:, safe_class].sum(), zt_s)[0]

                        g_harm = torch.zeros_like(g_safe)
                        for hc in harm_classes:
                            zt_h = prev_latents.clone().detach().requires_grad_(True)
                            z0_h = predict_z0(zt_h, noise_pred_uncond.detach(), alpha_bar_4d)
                            g_harm = g_harm + torch.autograd.grad(classifier(z0_h)[:, hc].sum(), zt_h)[0]

                        grad = (g_safe - args.harm_ratio * g_harm).detach()

                    # 5d. Attention-weighted guidance
                    weight = attn_weight * args.guidance_scale + \
                             (1.0 - attn_weight) * args.base_guidance_scale
                    grad_weighted = grad * weight
                    mask_ratio = (attn_weight > 0.3).float().mean().item()

                    if args.debug:
                        n_tokens = len(harmful_indices)
                        print(f"    [cross_attn] harmful_tokens={n_tokens} indices={harmful_indices} attn_mask_ratio={mask_ratio:.2%}")
                else:
                    # --- Original GradCAM-based guidance ---
                    grad_weighted, mask_ratio = spatial_guidance.compute_gradient(
                        classifier, z0_hat, prev_latents, noise_pred_uncond,
                        alpha_bar_4d, gradcam_map, spatial_thr,
                        args.guidance_scale, args.base_guidance_scale,
                        harm_ratio=args.harm_ratio,
                        soft=args.spatial_soft,
                    )

                # 6. Score-based guidance: adjust noise_pred and re-step
                denom = torch.sqrt(1 - alpha_bar)
                guidance_term = denom * grad_weighted

                # Clip guidance magnitude (0 = disable clipping)
                gt_norm = guidance_term.norm()
                eps_norm = noise_pred.norm()
                clipped = False
                if args.grad_clip_ratio > 0:
                    max_guidance = eps_norm * args.grad_clip_ratio
                    if gt_norm > max_guidance and gt_norm > 0:
                        guidance_term = guidance_term * (max_guidance / gt_norm)
                        clipped = True

                adjusted_noise_pred = noise_pred - guidance_term

                # Undo step index and re-step with adjusted noise
                if hasattr(pipe.scheduler, '_step_index') and \
                   pipe.scheduler._step_index is not None:
                    pipe.scheduler._step_index -= 1

                out = pipe.scheduler.step(
                    model_output=adjusted_noise_pred,
                    timestep=timestep,
                    sample=prev_latents,
                    return_dict=True,
                )
                callback_kwargs["latents"] = out.prev_sample

                if args.debug:
                    ratio = gt_norm / (eps_norm + 1e-8)
                    clipped_norm = guidance_term.norm().item()
                    print(f"    [guide] mask={mask_ratio:.2%} ||gt||={gt_norm:.4f} ratio={ratio:.2f} clip={clipped} final_gt={clipped_norm:.4f}")

                return callback_kwargs

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=[
                        "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
                    ],
                    num_images_per_prompt=args.nsamples,
                    height=512, width=512,
                )

            # Save image
            safe_prompt = "".join(
                c if c.isalnum() or c in ' -_' else '_' for c in prompt
            )[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(output.images[0], output_dir / img_filename)

            # Per-image stats
            guided_steps = monitor.stats["guided"]
            total_steps = monitor.stats["total"]
            guidance_ratio = guided_steps / max(total_steps, 1)

            # Extract per-step scores (dual mode uses different keys)
            if args.monitoring_mode == "dual":
                p_harm_history = [h.get("p_harm_cdf", h.get("p_harm", 0.0)) for h in monitor.stats["history"]]
            else:
                p_harm_history = [h["p_harm"] for h in monitor.stats["history"]]
            max_p_harm = max(p_harm_history) if p_harm_history else 0.0

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guided_steps": guided_steps,
                "skipped_steps": monitor.stats["skipped"],
                "total_steps": total_steps,
                "guidance_ratio": guidance_ratio,
                "max_p_harm": round(max_p_harm, 4),
                "p_harm_history": [round(p, 4) for p in p_harm_history],
            }
            all_stats.append(img_stats)

            print(f"  [{prompt_idx:03d}] Guided: {guided_steps}/{total_steps} "
                  f"({guidance_ratio*100:.1f}%)")

    # Summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0
    no_guidance = sum(1 for s in all_stats if s["guided_steps"] == 0)
    light = sum(1 for s in all_stats if 0 < s["guidance_ratio"] <= 0.3)
    medium = sum(1 for s in all_stats if 0.3 < s["guidance_ratio"] <= 0.7)
    heavy = sum(1 for s in all_stats if s["guidance_ratio"] > 0.7)

    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_guidance_ratio": float(avg_ratio),
            "no_guidance_count": no_guidance,
            "light_guidance_count": light,
            "medium_guidance_count": medium,
            "heavy_guidance_count": heavy,
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Avg guided: {avg_guided:.1f}/{args.num_inference_steps} ({avg_ratio*100:.1f}%)")
    print(f"Distribution: none={no_guidance}, light={light}, medium={medium}, heavy={heavy}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
