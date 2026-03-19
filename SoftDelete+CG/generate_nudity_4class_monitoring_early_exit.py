#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 4-Class: Sample-Level Monitoring + Spatial CG + Early Exit

Monitoring: P(harm) = CDF((heatmap.mean() - mu) / sigma)
  - Uses sample-level statistics (mean of heatmap distributions)
  - If P(harm) > monitoring_threshold → apply guidance

Early Exit: At first step, if max P(harm) < early_exit_threshold (default 0.3),
  skip ALL remaining guidance for this image.

Guidance: Spatial CG with pixel-level CDF normalization
  - Same as before, but only when monitoring triggers
"""

import os
import sys
import json
import math
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.distributions import Normal

import re
from transformers import CLIPTokenizer, CLIPTextModel

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


NUDITY_4CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


def load_gradcam_stats(stats_dir: str) -> Dict:
    """Load topk (for spatial) and sample-level (for monitoring) statistics."""
    stats_dir = Path(stats_dir)
    mapping = {2: "gradcam_stats_harm_nude_class2.json", 3: "gradcam_stats_harm_color_class3.json"}
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            # Top-K pixel stats (for spatial mask CDF normalization)
            topk = d.get("topk", {})
            # Sample-level stats (for monitoring P(harm))
            sample = d.get("sample_level", {})
            stats_map[cls] = {
                # Top-K pixels (for spatial masking) - use topk if available, fallback to all-pixel
                "topk_mean": float(topk.get("mean", d["mean"])),
                "topk_std": float(topk.get("std", d["std"])),
                # Sample-level (for monitoring P(harm))
                "sample_mean": float(sample.get("mean", d["mean"])),
                "sample_std": float(sample.get("std", d["std"])),
            }
    return stats_map


class SampleLevelMonitor:
    """
    Monitoring based on sample-level statistics.
    P(harm) = CDF((heatmap.mean() - mu) / sigma)
    """

    def __init__(self, classifier_model, stats_map: Dict, gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map
        self.gradcam = ClassifierGradCAM(classifier_model, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0), torch.tensor(1.0))

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        # Statistics tracking
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    def compute_p_harm(self, latent: torch.Tensor, timestep: torch.Tensor, harm_class: int,
                        topk_ratio: float = 0.2) -> float:
        """Compute P(harm) using top-k pixel statistics.

        Takes top `topk_ratio` (default 20%) of heatmap pixels,
        computes their mean, and compares against topk stats via CDF.
        """
        if harm_class not in self.stats_map:
            return 0.0

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        # Generate raw heatmap
        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(lat, norm_t, harm_class, normalize=False)

        # Take top 20% of heatmap pixels and compute their mean
        flat = heatmap.flatten()
        k = max(1, int(len(flat) * 0.1))
        topk_values = torch.topk(flat, k).values
        topk_mean = topk_values.mean().item()

        # Compare against topk stats
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]

        z = (topk_mean - mu) / (sigma + 1e-8)
        if math.isnan(z) or math.isinf(z):
            return 0.0, topk_mean, 0.0
        p_harm = self.normal.cdf(torch.tensor(z)).item()

        return p_harm, topk_mean, z

    def should_apply_guidance(self, latent: torch.Tensor, timestep: torch.Tensor,
                               monitoring_threshold: float, step: int) -> tuple:
        """
        Determine if guidance should be applied based on P(harm).
        Returns: (should_guide, active_harm_classes, info_dict)
        """
        self.stats["total_steps"] += 1

        active_classes = []
        info = {"step": step, "p_harm": {}, "heatmap_mean": {}, "z_score": {}}

        for harm_class in [2, 3]:
            if harm_class not in self.stats_map:
                continue

            p_harm, hm_mean, z = self.compute_p_harm(latent, timestep, harm_class)
            info["p_harm"][harm_class] = p_harm
            info["heatmap_mean"][harm_class] = hm_mean
            info["z_score"][harm_class] = z

            if p_harm > monitoring_threshold:
                active_classes.append(harm_class)

        info["active_classes"] = active_classes
        info["threshold"] = monitoring_threshold

        if len(active_classes) > 0:
            self.stats["guided_steps"] += 1
            self.stats["step_history"].append({**info, "guided": True})
            return True, active_classes, info
        else:
            self.stats["skipped_steps"] += 1
            self.stats["step_history"].append({**info, "guided": False})
            return False, [], info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}


class SpatialGuidance:
    """Spatial CG using top-K pixel CDF normalization for more accurate harmful region detection."""

    def __init__(self, classifier_model, stats_map: Dict, gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map
        self.gradcam = ClassifierGradCAM(classifier_model, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _pixel_cdf_normalize(self, heatmap: torch.Tensor, harm_class: int) -> torch.Tensor:
        """Apply CDF normalization using top-K pixel statistics (high-activation regions)."""
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def compute_gradient(self, latent: torch.Tensor, timestep: torch.Tensor,
                         active_harm_classes: list, spatial_threshold: float,
                         guidance_scale: float = 5.0, base_scale: float = 0.0) -> torch.Tensor:
        """Compute spatial guidance gradient."""
        if not active_harm_classes:
            return torch.zeros_like(latent)

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        # Generate masks for active harm classes
        masks = {}
        for hc in active_harm_classes:
            with torch.enable_grad():
                heatmap, _ = self.gradcam.generate_heatmap(lat, norm_t, hc, normalize=False)
            heatmap_norm = self._pixel_cdf_normalize(heatmap, hc)
            mask = (heatmap_norm >= spatial_threshold).float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            masks[hc] = mask

        # Compute gradients
        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in active_harm_classes:
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - g_harm

        # Combine masks
        combined_mask = None
        for hc in active_harm_classes:
            m = masks[hc]
            combined_mask = m if combined_mask is None else torch.max(combined_mask, m)

        if combined_mask is None:
            combined_mask = torch.zeros_like(latent[:, 0:1, :, :])

        # Apply spatial weighting
        weight = combined_mask * guidance_scale + (1 - combined_mask) * base_scale
        return (grad * weight).to(dtype=latent.dtype).detach()


def get_spatial_threshold(step: int, total: int, start: float, end: float, strategy: str = "cosine") -> float:
    """Compute spatial threshold based on step."""
    t = step / max(total - 1, 1)
    if strategy == "constant":
        return start
    elif strategy == "linear":
        return start - (start - end) * t
    elif strategy == "cosine":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/sample_level_monitoring")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)

    # Monitoring (sample-level)
    parser.add_argument("--monitoring_threshold", type=float, default=0.5,
                        help="P(harm) threshold for triggering guidance (0-1)")

    # Guidance (spatial)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--base_guidance_scale", type=float, default=2.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.5)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.1)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    parser.add_argument("--text_exit_threshold", type=float, default=0.0,
                        help="CLIP word-level cosine sim threshold. If max word sim < this, skip guidance entirely (0=disable)")
    parser.add_argument("--debug", action="store_true")

    # Prompt slicing for multi-GPU
    parser.add_argument("--start_idx", type=int, default=0, help="Start prompt index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End prompt index (exclusive), -1 for all")
    return parser.parse_args()


def load_prompts(f):
    """Load prompts from txt or CSV file.

    For CSV files: tries columns in priority order (matches SAFREE):
    - adv_prompt (MMA-Diffusion adversarial)
    - sensitive prompt (Ring-A-Bell)
    - prompt (I2P, UnlearnDiff, etc.)
    - target_prompt (MMA-Diffusion original)
    Returns: list of prompts
    """
    import csv
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames

            # Column priority (matches SAFREE: generate_safree.py)
            column_priority = [
                'adv_prompt',       # MMA-Diffusion adversarial
                'sensitive prompt', # Ring-A-Bell
                'prompt',           # I2P, UnlearnDiff, etc.
                'target_prompt',    # MMA-Diffusion original
                'text', 'Prompt', 'Text'  # fallbacks
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


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"SAMPLE-LEVEL MONITORING + SPATIAL CG")
    print(f"Monitoring threshold: {args.monitoring_threshold}")
    print(f"Guidance scale: {args.guidance_scale} (base: {args.base_guidance_scale})")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"{'='*60}\n")

    # Load stats
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)
    if not stats_map:
        raise RuntimeError(f"No stats found in {args.gradcam_stats_dir}")

    print("Loaded GradCAM stats:")
    for cls, s in stats_map.items():
        print(f"  Class {cls}:")
        print(f"    [Monitoring] sample_mean={s['sample_mean']:.4f}, sample_std={s['sample_std']:.4f}")
        print(f"    [Spatial]    topk_mean={s['topk_mean']:.4f}, topk_std={s['topk_std']:.4f}")

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()

    # Initialize modules
    monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    guidance = SpatialGuidance(classifier, stats_map, args.gradcam_layer, device)

    # Load CLIP for text-based early exit
    text_exit_sims = {}
    if args.text_exit_threshold > 0:
        print(f"\n[Text Early Exit] Loading CLIP for word-level similarity (threshold={args.text_exit_threshold})...")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

        # Get "nudity" reference embedding
        ref_tok = clip_tokenizer(["nudity"], return_tensors="pt").to(device)
        with torch.no_grad():
            ref_emb = clip_model(**ref_tok).pooler_output
        ref_emb = (ref_emb / ref_emb.norm(dim=-1, keepdim=True)).cpu()

        # Pre-compute word-level max sim for all prompts
        print(f"  Pre-computing word-level similarities for {len(prompts_with_idx)} prompts...")
        for pidx, prompt_text in prompts_with_idx:
            words = re.findall(r'[a-zA-Z]+', prompt_text)
            if not words:
                text_exit_sims[pidx] = (0.0, "")
                continue
            # Batch encode words
            word_toks = clip_tokenizer(words, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            with torch.no_grad():
                word_embs = clip_model(**word_toks).pooler_output.cpu()
            word_embs = word_embs / word_embs.norm(dim=-1, keepdim=True)
            sims = (word_embs @ ref_emb.T).squeeze()
            if sims.dim() == 0:
                sims = sims.unsqueeze(0)
            max_idx = sims.argmax().item()
            text_exit_sims[pidx] = (sims[max_idx].item(), words[max_idx])

        # Free CLIP memory
        del clip_model, clip_tokenizer, ref_emb
        torch.cuda.empty_cache()

        skipped = sum(1 for _, (sim, _) in text_exit_sims.items() if sim < args.text_exit_threshold)
        print(f"  Text early exit will skip {skipped}/{len(prompts_with_idx)} prompts")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        print(f"\n[Prompt {prompt_idx}/{len(all_prompts)}] {prompt}")
        for sample_idx in range(args.nsamples):
            # Use global seed for reproducibility (consistent with baseline evaluation)
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            monitor.reset_stats()
            early_exit_flag = [False]
            text_skipped = False

            # Text-based early exit: skip entire generation guidance if prompt is safe
            if args.text_exit_threshold > 0 and prompt_idx in text_exit_sims:
                sim_val, top_word = text_exit_sims[prompt_idx]
                if sim_val < args.text_exit_threshold:
                    early_exit_flag[0] = True
                    text_skipped = True
                    print(f"  [Text skip] sim={sim_val:.4f} top='{top_word}' < {args.text_exit_threshold} | {prompt[:60]}")
                else:
                    print(f"  [Text pass] sim={sim_val:.4f} top='{top_word}' >= {args.text_exit_threshold} | {prompt[:60]}")

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                if early_exit_flag[0]:
                    return callback_kwargs

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    # Monitoring: should we apply guidance?
                    should_guide, active_classes, info = monitor.should_apply_guidance(
                        latents, timestep, args.monitoring_threshold, step
                    )

                    if args.debug and step % 10 == 0:
                        p_harm_str = ", ".join([f"c{k}:{v:.3f}" for k, v in info["p_harm"].items()])
                        print(f"  Step {step}: P(harm)=[{p_harm_str}] -> {'GUIDE' if should_guide else 'skip'}")

                    if should_guide:
                        spatial_thr = get_spatial_threshold(
                            step, args.num_inference_steps,
                            args.spatial_threshold_start, args.spatial_threshold_end,
                            args.spatial_threshold_strategy
                        )
                        grad = guidance.compute_gradient(
                            latents, timestep, active_classes, spatial_thr,
                            args.guidance_scale, args.base_guidance_scale
                        )
                        callback_kwargs["latents"] = latents + grad

                return callback_kwargs

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )

            safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(output.images[0], output_dir / img_filename)

            # Compute per-class guidance stats
            class_guidance_counts = {2: 0, 3: 0}
            for step_info in monitor.stats["step_history"]:
                if step_info.get("guided", False):
                    for cls in step_info.get("active_classes", []):
                        class_guidance_counts[cls] = class_guidance_counts.get(cls, 0) + 1

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "early_exit": early_exit_flag[0],
                "text_skipped": text_skipped,
                "text_sim": text_exit_sims.get(prompt_idx, (0.0, ""))[0] if text_exit_sims else 0.0,
                "guided_steps": monitor.stats["guided_steps"],
                "skipped_steps": monitor.stats["skipped_steps"],
                "total_steps": monitor.stats["total_steps"],
                "guidance_ratio": monitor.stats["guided_steps"] / max(monitor.stats["total_steps"], 1),
                "class_guidance_counts": class_guidance_counts,
            }
            # Save detailed step history in debug mode
            if args.debug:
                img_stats["step_history"] = monitor.stats["step_history"]
            all_stats.append(img_stats)

            # Print per-image summary
            print(f"  [{prompt_idx:03d}] Guided: {img_stats['guided_steps']}/{img_stats['total_steps']} "
                  f"({img_stats['guidance_ratio']*100:.1f}%) | "
                  f"class2: {class_guidance_counts[2]}, class3: {class_guidance_counts[3]}")

    # Compute overall summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_skipped = np.mean([s["skipped_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0

    # Count images by guidance intensity
    no_guidance = sum(1 for s in all_stats if s["guided_steps"] == 0)
    light_guidance = sum(1 for s in all_stats if 0 < s["guidance_ratio"] <= 0.3)
    medium_guidance = sum(1 for s in all_stats if 0.3 < s["guidance_ratio"] <= 0.7)
    heavy_guidance = sum(1 for s in all_stats if s["guidance_ratio"] > 0.7)

    # Save summary stats
    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_skipped_steps": float(avg_skipped),
            "avg_guidance_ratio": float(avg_ratio),
            "no_guidance_count": no_guidance,
            "light_guidance_count": light_guidance,
            "medium_guidance_count": medium_guidance,
            "heavy_guidance_count": heavy_guidance,
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"\n[Guidance Statistics]")
    print(f"  Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps} ({avg_ratio*100:.1f}%)")
    print(f"  Avg skipped steps: {avg_skipped:.1f}/{args.num_inference_steps}")
    print(f"\n[Guidance Distribution]")
    print(f"  No guidance (0%):      {no_guidance:3d} images ({no_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Light (1-30%):         {light_guidance:3d} images ({light_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Medium (31-70%):       {medium_guidance:3d} images ({medium_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Heavy (71-100%):       {heavy_guidance:3d} images ({heavy_guidance/max(total_images,1)*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
