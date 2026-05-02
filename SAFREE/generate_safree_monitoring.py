#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE + Sample-Level Monitoring + Spatial CG

Uses the ORIGINAL ModifiedStableDiffusionPipeline for SAFREE (identical behavior),
then adds sample-level monitoring + spatial classifier guidance on top.

Architecture:
                    ┌─────────────────────────────────────────┐
    Prompt ────────►│ SAFREE Text Projection (ORIGINAL)       │
                    │ (Remove unsafe concept from embeddings) │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
    Latent ────────►│ Sample-Level Monitoring                 │
                    │ - P(harm) = CDF((mean - mu) / sigma)    │
                    │ - If P(harm) > threshold → Spatial CG   │
                    └─────────────────────────────────────────┘
"""

import os
import sys
import csv
import json
import math
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path

# Suppress tokenizer "Token indices sequence length is longer" warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from PIL import Image
from typing import List, Optional, Dict, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "SoftDelete+CG"))
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM

# Import original SAFREE pipeline
sys.path.insert(0, str(Path(__file__).parent))
from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline


# =============================================================================
# Extended Pipeline: Original SAFREE + Monitoring/CG callback
# =============================================================================

class MonitoringSaffreePipeline(ModifiedStableDiffusionPipeline):
    """
    Extends the original ModifiedStableDiffusionPipeline to add
    monitoring/CG after each scheduler step. SAFREE logic is 100% inherited.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        negative_prompt_space=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        prompt_ids=None,
        prompt_embeddings=None,
        return_latents=False,
        safree_dict={},
        # NEW: monitoring/CG callback
        monitoring_callback=None,
    ):
        from models.modified_stable_diffusion_pipeline import (
            projection_matrix, safree_projection, projection_and_orthogonal, f_beta
        )

        # 0) defaults
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        sf = safree_dict or {}
        sf.setdefault("safree", False)
        sf.setdefault("svf", False)
        sf.setdefault("lra", False)
        sf.setdefault("alpha", 0.0)
        sf.setdefault("re_attn_t", [-1, 10000])
        sf.setdefault("up_t", 10)
        sf.setdefault("category", "nudity")
        sf.setdefault("logger", None)

        # 1) inputs check
        self.check_inputs(prompt, height, width, callback_steps, prompt_embeds=prompt_embeddings)

        # 2) flags
        batch_size = 1
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3) encode prompt (ORIGINAL)
        text_embeddings, text_input_ids, attention_mask = self._new_encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_ids, prompt_embeddings
        )

        # projection matrices (ORIGINAL)
        negspace_tokens = negative_prompt_space if negative_prompt_space is not None else (negative_prompt or "")
        negspace_text_embeddings = self._new_encode_negative_prompt_space(
            negspace_tokens, max_length=self.tokenizer.model_max_length, num_images_per_prompt=num_images_per_prompt
        )
        project_matrix_mat = projection_matrix(negspace_text_embeddings.T)

        masked_embs = self._masked_encode_prompt(prompt)
        masked_project_matrix = projection_matrix(masked_embs.T)

        # SAFREE projection (ORIGINAL)
        if sf.get("safree", False):
            rescaled_text_embeddings = safree_projection(
                text_embeddings,
                masked_embs,
                masked_project_matrix,
                project_matrix_mat,
                alpha=sf["alpha"],
                max_length=self.tokenizer.model_max_length,
                logger=sf.get("logger", None),
            )
        else:
            rescaled_text_embeddings = text_embeddings

        # SVF (ORIGINAL)
        if sf.get("svf", False):
            proj_ort = projection_and_orthogonal(text_embeddings, masked_project_matrix, project_matrix_mat)

            _, text_e = text_embeddings.chunk(2)
            s_attn_mask = (attention_mask.squeeze() == 1) if (attention_mask is not None) else torch.ones(
                text_e.shape[1], dtype=torch.bool, device=text_e.device
            )

            text_e = text_e.squeeze(0)
            _, proj_ort_e = proj_ort.chunk(2)
            proj_ort_e = proj_ort_e.squeeze(0)

            proj_ort_e_act = proj_ort_e[s_attn_mask]
            text_e_act = text_e[s_attn_mask]

            sim_org_onp_act = F.cosine_similarity(proj_ort_e_act.float(), text_e_act.float(), dim=-1)
            if torch.isnan(sim_org_onp_act).any():
                sim_org_onp_act = torch.nan_to_num(sim_org_onp_act, nan=0.0)

            beta = (1 - sim_org_onp_act.mean().item())
            beta_adjusted = f_beta(beta, upperbound_timestep=sf["up_t"], concept_type=sf["category"])

            # Safeguard: beta ≈ 1.0 means projection collapsed (common with long prompts > 77 tokens)
            if beta > 0.95:
                print(f"[SAFREE WARNING] beta={beta:.4f} > 0.95 — projection unstable. "
                      f"Falling back to original embeddings.")
                beta_adjusted = -1

            # Also check rescaled embeddings for NaN
            if torch.isnan(rescaled_text_embeddings).any():
                print("[SAFREE WARNING] NaN in rescaled embeddings — falling back to original.")
                rescaled_text_embeddings = text_embeddings
                beta_adjusted = -1

            if sf.get("logger", None) is not None:
                sf["logger"].log(f"beta : {beta}, adjusted_beta: {beta_adjusted}")
            else:
                print(f"[SAFREE SVF] beta={beta:.4f}, adjusted_beta={beta_adjusted}")
        else:
            beta_adjusted = -1

        # 4) timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5) latents
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6) extra kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7) denoising loop (ORIGINAL + monitoring/CG)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand latents (ORIGINAL)
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # choose embeddings (ORIGINAL SVF step-gating)
                if sf.get("svf", False):
                    use_rescaled = sf.get("safree", False) and (i <= beta_adjusted)
                else:
                    lo, hi = sf.get("re_attn_t", [-1, 1000000])
                    use_rescaled = sf.get("safree", False) and (lo <= i <= hi)

                _text_embeddings = rescaled_text_embeddings if use_rescaled else text_embeddings

                # unet (ORIGINAL)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=_text_embeddings).sample

                # CFG (ORIGINAL)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # step (ORIGINAL)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # === NEW: Monitoring + Spatial CG ===
                if monitoring_callback is not None:
                    latents = monitoring_callback(i, t, latents)

                # progress (ORIGINAL)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % (callback_steps or 1) == 0:
                        callback(i, t, latents)

        if return_latents:
            return latents

        # 8) decode (ORIGINAL)
        image = self.decode_latents(latents)

        # 10) to PIL (ORIGINAL)
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image


# =============================================================================
# Classifier Config
# =============================================================================

NUDITY_4CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


# =============================================================================
# Monitoring & Guidance Classes
# =============================================================================

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
            topk = d.get("topk", {})
            sample = d.get("sample_level", {})
            stats_map[cls] = {
                "topk_mean": float(topk.get("mean", d["mean"])),
                "topk_std": float(topk.get("std", d["std"])),
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

        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    def compute_p_harm(self, latent: torch.Tensor, timestep: torch.Tensor, harm_class: int) -> tuple:
        if harm_class not in self.stats_map:
            return 0.0, 0.0, 0.0

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(lat, norm_t, harm_class, normalize=False)

        heatmap_mean = heatmap.mean().item()
        mu = self.stats_map[harm_class]["sample_mean"]
        sigma = self.stats_map[harm_class]["sample_std"]

        z = (heatmap_mean - mu) / (sigma + 1e-8)
        if math.isnan(z) or math.isinf(z):
            return 0.0, heatmap_mean, 0.0
        p_harm = self.normal.cdf(torch.tensor(z)).item()

        return p_harm, heatmap_mean, z

    def should_apply_guidance(self, latent: torch.Tensor, timestep: torch.Tensor,
                               monitoring_threshold: float, step: int) -> tuple:
        self.stats["total_steps"] += 1

        active_classes = []
        info = {"step": step, "p_harm": {}, "heatmap_mean": {}, "z_score": {}}

        for harm_class in [2]:  # Only harm_nude for monitoring
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
    """Spatial CG using top-K pixel CDF normalization."""

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
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def compute_gradient(self, latent: torch.Tensor, timestep: torch.Tensor,
                         active_harm_classes: list, spatial_threshold: float,
                         guidance_scale: float = 5.0, base_scale: float = 0.0,
                         harmful_scale: float = 1.0) -> torch.Tensor:
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
            for hc in active_harm_classes:
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
# Utilities (same as original gen_safree_single.py)
# =============================================================================

def get_negative_prompt_space(category: str) -> List[str]:
    if category == "nudity" or category == "sexual":
        return [
            "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
            "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
            "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
            "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
        ]
    elif category == "violence":
        return [
            "violence", "assault", "attack", "fight", "beating", "punching",
            "kicking", "hitting", "stabbing", "knife attack", "shooting", "gun violence",
            "aiming a gun", "strangling", "choking", "execution", "blood", "bloody scene",
            "open wound", "corpse", "explosion", "grenade blast", "bomb blast",
            "molotov cocktail", "arson"
        ]
    else:
        return []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = ArgumentParser(description="SAFREE + Sample-Level Monitoring + Spatial CG")

    # Model & Generation
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/safree_monitoring")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0, help="Start prompt index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End prompt index (exclusive), -1 for all")

    # SAFREE parameters
    parser.add_argument("--safree", action="store_true", help="Enable SAFREE text projection")
    parser.add_argument("--safree_alpha", type=float, default=0.01)
    parser.add_argument("--svf", action="store_true", help="Enable Self-Validation Filter")
    parser.add_argument("--svf_up_t", type=int, default=10)
    parser.add_argument("--category", type=str, default="nudity",
                        choices=["nudity", "sexual", "violence"])

    # Classifier (single 4-class)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")

    # Monitoring (sample-level P(harm) threshold)
    parser.add_argument("--monitoring_threshold", type=float, default=0.5,
                        help="P(harm) threshold for triggering guidance (0-1)")

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

    # Text-based early exit
    parser.add_argument("--text_exit_threshold", type=float, default=0.0,
                        help="CLIP word-level cosine sim threshold. Below = skip CG (SAFREE still runs). 0=disable")

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
    print(f"SAFREE + SAMPLE-LEVEL MONITORING + SPATIAL CG")
    print(f"{'='*70}")
    print(f"SAFREE: {args.safree} (alpha={args.safree_alpha}, SVF={args.svf})")
    print(f"Classifier: {args.classifier_ckpt}")
    print(f"Monitoring threshold: {args.monitoring_threshold}")
    print(f"Guidance scale: {args.guidance_scale} (base: {args.base_guidance_scale})")
    print(f"Harmful scale: {args.harmful_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"SEED: {args.seed} (FIXED for all samples)")
    print(f"{'='*70}\n")

    # Load stats
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)
    if not stats_map:
        raise RuntimeError(f"No stats found in {args.gradcam_stats_dir}")

    print("Loaded GradCAM stats:")
    for cls, s in stats_map.items():
        print(f"  Class {cls}: topk_mean={s['topk_mean']:.4f}, sample_mean={s['sample_mean']:.4f}")

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    start_idx = args.start_idx
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"\nLoaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load pipeline (ORIGINAL ModifiedStableDiffusionPipeline with DPMSolver)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.ckpt_path, subfolder="scheduler")
    pipe = MonitoringSaffreePipeline.from_pretrained(
        args.ckpt_path,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # Load classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True,
        channel=4, num_classes=4
    ).to(device)
    classifier.eval()
    print(f"Loaded 4-class classifier")

    # Initialize modules
    monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    guidance_module = SpatialGuidance(classifier, stats_map, args.gradcam_layer, device)

    # SAFREE setup
    negative_prompt_space = get_negative_prompt_space(args.category) if args.safree else []
    negative_prompt = ", ".join(negative_prompt_space) if negative_prompt_space else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Text-based early exit: pre-compute word-level CLIP cosine sim ===
    import re
    text_exit_sims = {}
    if args.text_exit_threshold > 0:
        from transformers import CLIPTokenizer, CLIPTextModel
        print("\n[Text Exit] Loading CLIP for word-level cosine sim pre-computation...")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

        ref_tok = clip_tokenizer(["nudity"], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ref_emb = clip_model(**ref_tok).pooler_output
        ref_emb = (ref_emb / ref_emb.norm(dim=-1, keepdim=True)).cpu()

        for pidx, prompt_text in tqdm(prompts_with_idx, desc="[Text Exit] Pre-computing"):
            words = re.findall(r'[a-zA-Z]+', prompt_text)
            if not words:
                text_exit_sims[pidx] = (0.0, "")
                continue
            word_toks = clip_tokenizer(words, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                word_embs = clip_model(**word_toks).pooler_output.cpu()
            word_embs = word_embs / word_embs.norm(dim=-1, keepdim=True)
            sims = (word_embs @ ref_emb.T).squeeze(-1)
            max_idx = sims.argmax().item()
            text_exit_sims[pidx] = (sims[max_idx].item(), words[max_idx])

        del clip_model, clip_tokenizer, ref_emb
        torch.cuda.empty_cache()

        skip_count = sum(1 for v in text_exit_sims.values() if v[0] < args.text_exit_threshold)
        print(f"[Text Exit] threshold={args.text_exit_threshold}, will skip CG for {skip_count}/{len(text_exit_sims)} prompts")

    all_stats = []
    gen = torch.Generator(device=device)

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            monitor.reset_stats()
            text_skipped = False

            # Check text-based early exit
            if args.text_exit_threshold > 0 and prompt_idx in text_exit_sims:
                sim_val, top_word = text_exit_sims[prompt_idx]
                if sim_val < args.text_exit_threshold:
                    text_skipped = True
                    print(f"  [Text skip] sim={sim_val:.4f} top='{top_word}' < {args.text_exit_threshold} | {prompt[:60]}")
                else:
                    print(f"  [Text pass] sim={sim_val:.4f} top='{top_word}' >= {args.text_exit_threshold} | {prompt[:60]}")

            # Build monitoring callback (closure captures current state)
            def make_monitoring_callback(skip_cg=False):
                def monitoring_callback(step_idx, t, latents):
                    if skip_cg:
                        return latents
                    if args.guidance_start_step <= step_idx <= args.guidance_end_step:
                        should_guide, active_classes, info = monitor.should_apply_guidance(
                            latents, t, args.monitoring_threshold, step_idx
                        )

                        if args.debug and step_idx % 10 == 0:
                            p_harm = info["p_harm"].get(2, 0)
                            print(f"  Step {step_idx:2d}: P(harm)={p_harm:.3f} -> {'GUIDE' if should_guide else 'skip'}")

                        if should_guide:
                            spatial_thr = get_spatial_threshold(
                                step_idx, args.num_inference_steps,
                                args.spatial_threshold_start, args.spatial_threshold_end,
                                args.spatial_threshold_strategy
                            )
                            grad = guidance_module.compute_gradient(
                                latents, t, active_classes, spatial_thr,
                                args.guidance_scale, args.base_guidance_scale,
                                args.harmful_scale
                            )
                            return latents + grad
                    return latents
                return monitoring_callback

            # Generate using ORIGINAL SAFREE pipeline + monitoring callback
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
                monitoring_callback=make_monitoring_callback(skip_cg=text_skipped),
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
                "monitoring_threshold": args.monitoring_threshold,
                "text_skipped": text_skipped,
                "text_sim": text_exit_sims.get(prompt_idx, (0.0, ""))[0] if args.text_exit_threshold > 0 else None
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
