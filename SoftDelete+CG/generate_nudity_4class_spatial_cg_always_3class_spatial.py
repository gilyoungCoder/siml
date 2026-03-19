#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 4-Class: safe-harm_nude-harm_color ALL spatial guidance (NO restricted)

Applies spatial guidance to ALL 3 classes simultaneously:
  - safe (class 1): pull toward
  - harm_nude (class 2): push away
  - harm_color (class 3): push away

NO restricted gradient - just simple combination:
  grad = grad_safe - harmful_scale * (grad_nude + grad_color)

Each class has its own spatial mask from GradCAM.
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


NUDITY_4CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


class AdaptiveSpatialThresholdScheduler:
    def __init__(self, strategy="linear_decrease", start_value=0.7, end_value=0.3, total_steps=50):
        self.strategy, self.start_value, self.end_value, self.total_steps = strategy, start_value, end_value, total_steps

    def get_threshold(self, current_step):
        if self.strategy == "constant": return self.start_value
        t = current_step / max(self.total_steps - 1, 1)
        if self.strategy == "linear_decrease": return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "cosine_anneal": return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        return self.start_value


def load_gradcam_stats_map(stats_dir):
    stats_dir = Path(stats_dir)
    mapping = {2: "gradcam_stats_harm_nude_class2.json", 3: "gradcam_stats_harm_color_class3.json"}
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f: d = json.load(f)
            stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
    return stats_map


class ThreeClassSpatialMaskGenerator:
    """Generate spatial masks for all 3 classes (safe, harm_nude, harm_color)."""

    def __init__(self, classifier_model, gradcam_layer="encoder_model.middle_block.2",
                 device="cuda", debug=False, gradcam_stats_map=None):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.gradcam_stats_map = gradcam_stats_map
        self.gradcam = ClassifierGradCAM(classifier_model=classifier_model, target_layer_name=gradcam_layer)
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)
        self.stats = {'total_steps': 0, 'step_history': []}

    def _apply_cdf_normalization(self, heatmap, mean, std):
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        normal = Normal(torch.tensor(0.0, device=heatmap.device), torch.tensor(1.0, device=heatmap.device))
        return normal.cdf(z)

    def _generate_heatmap(self, latent, norm_timestep, target_class):
        use_abs = self.gradcam_stats_map and target_class in self.gradcam_stats_map
        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(latent=latent, timestep=norm_timestep,
                                                        target_class=target_class, normalize=not use_abs)
        if use_abs:
            stats = self.gradcam_stats_map[target_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])
        return heatmap

    def generate_masks(self, latent, timestep, spatial_threshold, current_step=None):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != latent.shape[0]:
            timestep = timestep.expand(latent.shape[0])

        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        # Generate masks for harm_nude and harm_color
        masks_dict = {}
        for harm_class in [2, 3]:
            heatmap = self._generate_heatmap(latent_input, norm_timestep, harm_class)
            mask = (heatmap >= spatial_threshold).float()
            masks_dict[harm_class] = mask

        self.stats['total_steps'] += 1
        if self.debug:
            print(f"  [Step {current_step}] Masks generated")

        return masks_dict

    def reset_statistics(self):
        self.stats = {'total_steps': 0, 'step_history': []}


class ThreeClassSpatialGuidance:
    """
    Simple 3-class spatial guidance (NO restricted gradient).
    grad = grad_safe - harmful_scale * (grad_nude + grad_color)
    """

    def __init__(self, classifier_model, device="cuda"):
        self.classifier = classifier_model.to(device)
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _compute_class_gradient(self, latent, norm_timestep, target_class):
        latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)
        logits = self.classifier(latent_input, norm_timestep)
        grad = torch.autograd.grad(logits[:, target_class].sum(), latent_input)[0]
        return grad

    def compute_gradient(self, latent, timestep, masks_dict, guidance_scale=5.0,
                         harmful_scale=1.0, base_guidance_scale=0.0):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).to(latent.device)
        if timestep.shape[0] != latent.shape[0]:
            timestep = timestep.expand(latent.shape[0]).to(latent.device)
        norm_timestep = timestep.float() / 1000.0

        with torch.enable_grad():
            grad_safe = self._compute_class_gradient(latent, norm_timestep, 1)  # safe_clothed
            grad_nude = self._compute_class_gradient(latent, norm_timestep, 2)  # harm_nude
            grad_color = self._compute_class_gradient(latent, norm_timestep, 3) # harm_color

            # Simple combination (no restricted)
            grad = grad_safe - harmful_scale * (grad_nude + grad_color)

        # Union mask
        mask_nude = masks_dict.get(2, torch.zeros_like(latent[:, 0:1, :, :]))
        mask_color = masks_dict.get(3, torch.zeros_like(latent[:, 0:1, :, :]))
        if mask_nude.dim() == 3: mask_nude = mask_nude.unsqueeze(1)
        if mask_color.dim() == 3: mask_color = mask_color.unsqueeze(1)
        combined_mask = torch.max(mask_nude, mask_color)

        weight_map = combined_mask * guidance_scale + (1 - combined_mask) * base_guidance_scale
        return (grad * weight_map).to(dtype=latent.dtype).detach()

    def apply_guidance(self, latent, timestep, masks_dict, guidance_scale=5.0,
                       harmful_scale=1.0, base_guidance_scale=0.0):
        return latent + self.compute_gradient(latent, timestep, masks_dict, guidance_scale,
                                               harmful_scale, base_guidance_scale)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3)
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease")
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--base_guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_prompts(f): return [l.strip() for l in open(f) if l.strip()]
def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray): img = Image.fromarray(img)
    img.resize((512, 512)).save(path)
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}\n3-CLASS SPATIAL GUIDANCE (NO RESTRICTED)\n{'='*60}")

    gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir) if args.gradcam_stats_dir else None
    prompts = load_prompts(args.prompt_file)

    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    classifier = load_discriminator(ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4).to(device)
    classifier.eval()

    threshold_scheduler = AdaptiveSpatialThresholdScheduler(args.threshold_strategy, args.spatial_threshold_start,
                                                             args.spatial_threshold_end, args.num_inference_steps)
    mask_generator = ThreeClassSpatialMaskGenerator(classifier, args.gradcam_layer, device, args.debug, gradcam_stats_map)
    guidance_module = ThreeClassSpatialGuidance(classifier, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        for sample_idx in range(args.nsamples):
            mask_generator.reset_statistics()

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                if args.guidance_start_step <= step <= args.guidance_end_step:
                    spatial_threshold = threshold_scheduler.get_threshold(step)
                    masks_dict = mask_generator.generate_masks(latents, timestep, spatial_threshold, step)
                    guided_latents = guidance_module.apply_guidance(
                        latents, timestep, masks_dict, args.guidance_scale, args.harmful_scale, args.base_guidance_scale)
                    callback_kwargs["latents"] = guided_latents
                return callback_kwargs

            with torch.no_grad():
                output = pipe(prompt=prompt, num_inference_steps=args.num_inference_steps,
                              guidance_scale=args.cfg_scale, callback_on_step_end=callback_on_step_end,
                              callback_on_step_end_tensor_inputs=["latents"])

            safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            save_image(output.images[0], output_dir / f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
