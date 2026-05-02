#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE + Dual Classifier Guidance

Uses the ORIGINAL ModifiedStableDiffusionPipeline for SAFREE (identical behavior),
then adds dual classifier guidance (3-class monitoring + 4-class spatial CG) on top.

Architecture:
                    ┌─────────────────────────────────────────┐
    Prompt ────────►│ SAFREE Text Projection (ORIGINAL)       │
                    │ (Remove unsafe concept from embeddings) │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
    Latent ────────►│ Dual Classifier Guidance                │
                    │ - 3-class: Monitor (guide if harmful)   │
                    │ - 4-class: GradCAM mask + gradient      │
                    └─────────────────────────────────────────┘
"""

import os
import sys
import csv
import json
import math
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal

from diffusers.schedulers import DPMSolverMultistepScheduler

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "SoftDelete+CG"))
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM

# Import original SAFREE pipeline (subclassed with monitoring callback)
sys.path.insert(0, str(Path(__file__).parent))
from generate_safree_monitoring import MonitoringSaffreePipeline, get_negative_prompt_space, set_seed, load_prompts, save_image


# =============================================================================
# Classifier Configs
# =============================================================================

THREE_CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2,
    "safe_classes": [0, 1], "harm_classes": [2],
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude"}
}

FOUR_CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


# =============================================================================
# Dual Classifier Classes
# =============================================================================

def load_gradcam_stats(stats_dir: str) -> Dict:
    """Load topk and sample-level statistics for 4-class (guidance)."""
    stats_dir = Path(stats_dir)
    mapping = {2: "gradcam_stats_harm_nude_class2.json", 3: "gradcam_stats_harm_color_class3.json"}
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            topk = d.get("topk", {})
            sample = d.get("sample_level", {})
            stats_map[cls] = {
                "topk_mean": float(topk.get("mean", d["mean"])),
                "topk_std": float(topk.get("std", d["std"])),
                "sample_mean": float(sample.get("mean", d["mean"])),
                "sample_std": float(sample.get("std", d["std"])),
            }
    return stats_map


class DualClassifierMonitor:
    """
    Monitoring using 3-class classifier with argmax-based decision.
    Guide if argmax(logits) == 2 (classifier predicts harm).
    """

    def __init__(self, classifier_3class, device: str = "cuda"):
        self.classifier = classifier_3class.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        self.reset_state()

    def reset_state(self):
        self.stats = {
            "total_steps": 0,
            "guided_steps": 0,
            "skipped_steps": 0,
            "step_history": []
        }

    def should_apply_guidance(self, latent: torch.Tensor, timestep: torch.Tensor, step: int) -> tuple:
        self.stats["total_steps"] += 1

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.no_grad():
            logits = self.classifier(lat, norm_t)
            probs = F.softmax(logits, dim=-1)
            p_harm = probs[0, 2].item()

        pred_class = int(logits[0].argmax().item())

        info = {
            "step": step,
            "p_harm": p_harm,
            "logits": logits[0].cpu().tolist(),
            "pred_class": pred_class,
            "class_name": {0: "benign", 1: "safe", 2: "harm"}[pred_class]
        }

        should_guide = (pred_class == 2)

        if should_guide:
            self.stats["guided_steps"] += 1
            self.stats["step_history"].append({**info, "guided": True})
        else:
            self.stats["skipped_steps"] += 1
            self.stats["step_history"].append({**info, "guided": False})

        return should_guide, info


class SpatialGuidance4Class:
    """
    Spatial CG using 4-class classifier for gradient computation.
    Gradient: g_safe - harmful_scale * g_harm
    """

    def __init__(self, classifier_4class, stats_map_4class: Dict,
                 gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_4class.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map_4class
        self.gradcam = ClassifierGradCAM(classifier_4class, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _pixel_cdf_normalize(self, heatmap: torch.Tensor, harm_class: int) -> torch.Tensor:
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def compute_gradient(self, latent: torch.Tensor, timestep: torch.Tensor,
                         spatial_threshold: float,
                         guidance_scale: float = 5.0, base_scale: float = 0.0,
                         harmful_scale: float = 1.0) -> torch.Tensor:
        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        # Generate masks for harm class (nude only)
        harm_classes = [2]
        masks = {}
        for hc in harm_classes:
            if hc not in self.stats_map:
                continue
            with torch.enable_grad():
                heatmap, _ = self.gradcam.generate_heatmap(lat, norm_t, hc, normalize=False)
            heatmap_norm = self._pixel_cdf_normalize(heatmap, hc)
            mask = (heatmap_norm >= spatial_threshold).float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            masks[hc] = mask

        # Compute gradients: g_safe - harmful_scale * g_harm
        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in harm_classes:
                if hc not in self.stats_map:
                    continue
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - harmful_scale * g_harm

        # Combine masks
        combined_mask = None
        for hc in masks:
            m = masks[hc]
            combined_mask = m if combined_mask is None else torch.max(combined_mask, m)

        if combined_mask is None:
            combined_mask = torch.zeros_like(latent[:, 0:1, :, :])

        # Apply spatial weighting
        weight = combined_mask * guidance_scale + (1 - combined_mask) * base_scale
        return (grad * weight).to(dtype=latent.dtype).detach()


def get_spatial_threshold(step: int, total: int, start: float, end: float, strategy: str = "cosine") -> float:
    t = step / max(total - 1, 1)
    if strategy == "constant":
        return start
    elif strategy == "linear":
        return start - (start - end) * t
    elif strategy == "cosine":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = ArgumentParser(description="SAFREE + Dual Classifier Guidance")

    # Model & Generation
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/safree_dual_classifier")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # SAFREE parameters
    parser.add_argument("--safree", action="store_true", help="Enable SAFREE text projection")
    parser.add_argument("--safree_alpha", type=float, default=0.01)
    parser.add_argument("--svf", action="store_true", help="Enable Self-Validation Filter")
    parser.add_argument("--svf_up_t", type=int, default=10)
    parser.add_argument("--category", type=str, default="nudity",
                        choices=["nudity", "sexual", "violence"])

    # 3-class classifier (monitoring)
    parser.add_argument("--classifier_3class_ckpt", type=str, required=True)

    # 4-class classifier (guidance)
    parser.add_argument("--classifier_4class_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")

    # Guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--base_guidance_scale", type=float, default=2.0)
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.5)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.1)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Fixed seed for ALL generations
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\n{'='*70}")
    print(f"SAFREE + DUAL CLASSIFIER GUIDANCE")
    print(f"{'='*70}")
    print(f"SAFREE: {args.safree} (alpha={args.safree_alpha}, SVF={args.svf})")
    print(f"3-class (monitoring): {args.classifier_3class_ckpt}")
    print(f"4-class (guidance):   {args.classifier_4class_ckpt}")
    print(f"Guidance scale: {args.guidance_scale} (base: {args.base_guidance_scale})")
    print(f"Harmful scale: {args.harmful_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"SEED: {args.seed} (FIXED for all samples)")
    print(f"{'='*70}\n")

    # Load stats for 4-class
    stats_4class = load_gradcam_stats(args.gradcam_stats_dir)
    if not stats_4class:
        raise RuntimeError(f"No 4-class stats found in {args.gradcam_stats_dir}")

    print("Loaded 4-class GradCAM stats:")
    for cls, s in stats_4class.items():
        print(f"  Class {cls}: topk_mean={s['topk_mean']:.4f}, topk_std={s['topk_std']:.4f}")

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"\nLoaded {len(prompts)} prompts")

    # Load pipeline (ORIGINAL SAFREE with DPMSolver)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.ckpt_path, subfolder="scheduler")
    pipe = MonitoringSaffreePipeline.from_pretrained(
        args.ckpt_path,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # Load classifiers
    classifier_3class = load_discriminator(
        ckpt_path=args.classifier_3class_ckpt, condition=None, eval=True,
        channel=4, num_classes=3
    ).to(device)
    classifier_3class.eval()
    print(f"Loaded 3-class classifier for monitoring")

    classifier_4class = load_discriminator(
        ckpt_path=args.classifier_4class_ckpt, condition=None, eval=True,
        channel=4, num_classes=4
    ).to(device)
    classifier_4class.eval()
    print(f"Loaded 4-class classifier for guidance")

    # Initialize modules
    monitor = DualClassifierMonitor(classifier_3class, device)
    guidance = SpatialGuidance4Class(classifier_4class, stats_4class, args.gradcam_layer, device)

    # SAFREE setup
    negative_prompt_space = get_negative_prompt_space(args.category) if args.safree else []
    negative_prompt = ", ".join(negative_prompt_space) if negative_prompt_space else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    gen = torch.Generator(device=device)

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        for sample_idx in range(args.nsamples):
            monitor.reset_state()

            # Build monitoring callback (closure captures current state)
            def make_monitoring_callback():
                def monitoring_callback(step_idx, t, latents):
                    if args.guidance_start_step <= step_idx <= args.guidance_end_step:
                        should_guide, info = monitor.should_apply_guidance(latents, t, step_idx)

                        if args.debug and step_idx % 10 == 0:
                            pred = info["class_name"]
                            p_harm = info["p_harm"]
                            print(f"  Step {step_idx:2d}: pred={pred}, P(harm)={p_harm:.3f} -> {'GUIDE' if should_guide else 'skip'}")

                        if should_guide:
                            spatial_thr = get_spatial_threshold(
                                step_idx, args.num_inference_steps,
                                args.spatial_threshold_start, args.spatial_threshold_end,
                                args.spatial_threshold_strategy
                            )
                            grad = guidance.compute_gradient(
                                latents, t, spatial_thr,
                                args.guidance_scale, args.base_guidance_scale,
                                args.harmful_scale
                            )
                            return latents + grad
                    return latents
                return monitoring_callback

            # Generate using ORIGINAL SAFREE pipeline + dual classifier callback
            imgs = pipe(
                prompt,
                num_images_per_prompt=args.nsamples,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                height=512,
                width=512,
                generator=gen.manual_seed(args.seed),
                safree_dict={
                    "safree": args.safree,
                    "svf": args.svf,
                    "alpha": args.safree_alpha,
                    "up_t": args.svf_up_t,
                    "category": args.category,
                    "re_attn_t": [-1, 10000],
                },
                monitoring_callback=make_monitoring_callback(),
            )

            # Save
            safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"

            img = imgs[0] if isinstance(imgs, list) else imgs
            if isinstance(img, Image.Image):
                save_image(img, output_dir / img_filename)
            else:
                save_image(img, output_dir / img_filename)

            # Record stats
            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": args.seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guided_steps": monitor.stats["guided_steps"],
                "skipped_steps": monitor.stats["skipped_steps"],
                "total_steps": monitor.stats["total_steps"],
                "guidance_ratio": monitor.stats["guided_steps"] / max(monitor.stats["total_steps"], 1),
                "safree_enabled": args.safree,
                "svf_beta_adjusted": -1,  # tracked inside pipeline
            }
            if args.debug:
                img_stats["step_history"] = monitor.stats["step_history"]
            all_stats.append(img_stats)

            print(f"  [{prompt_idx:03d}] Guided: {img_stats['guided_steps']}/{img_stats['total_steps']} "
                  f"({img_stats['guidance_ratio']*100:.1f}%)")

    # Summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0

    no_guidance = sum(1 for s in all_stats if s["guided_steps"] == 0)
    light_guidance = sum(1 for s in all_stats if 0 < s["guidance_ratio"] <= 0.3)
    medium_guidance = sum(1 for s in all_stats if 0.3 < s["guidance_ratio"] <= 0.7)
    heavy_guidance = sum(1 for s in all_stats if s["guidance_ratio"] > 0.7)

    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": avg_guided,
            "avg_guidance_ratio": avg_ratio,
            "no_guidance_count": no_guidance,
            "light_guidance_count": light_guidance,
            "medium_guidance_count": medium_guidance,
            "heavy_guidance_count": heavy_guidance,
        },
        "samples": all_stats
    }

    stats_file = output_dir / "generation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total images: {total_images}")
    print(f"Avg guided steps: {avg_guided:.1f}")
    print(f"Avg guidance ratio: {avg_ratio*100:.1f}%")
    print(f"No guidance: {no_guidance}, Light: {light_guidance}, Medium: {medium_guidance}, Heavy: {heavy_guidance}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
