#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE++ Spatial Classifier Guidance

Combines two complementary safety mechanisms:
1. SAFREE (Text-level): Projects text embeddings to remove unsafe concept subspace
2. Spatial CG (Image-level): Applies classifier guidance to harmful regions

Key Features:
- Text-level projection removes unsafe tokens from prompt embeddings
- Image-level classifier detects harmful content at each step
- GradCAM-based spatial masking targets only harmful regions
- Bidirectional guidance: safe_grad - harm_grad in harmful regions

Architecture:
                    ┌─────────────────────────────────────────┐
    Prompt ────────►│ SAFREE Text Projection                  │
                    │ (Remove unsafe concept from embeddings) │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
    Latent ────────►│ Spatial Classifier Guidance             │
                    │ - Detect harmful regions (4-class)      │
                    │ - GradCAM spatial mask                  │
                    │ - Bidirectional gradient guidance       │
                    └─────────────────────────────────────────┘
"""

import os
import sys
import json
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from transformers.modeling_outputs import BaseModelOutputWithPooling
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "SoftDelete+CG"))

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM

logger = logging.get_logger(__name__)


# =========================
# SAFREE Helper Functions
# =========================
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def f_beta(z: float, btype: str = "sigmoid", upperbound_timestep: int = 10, concept_type: str = "nudity") -> int:
    """Dynamic beta calculation for Self-Validation Filter."""
    if "vangogh" in concept_type:
        t = 5.5
        k = 3.5
    else:
        t = 5.333
        k = 2.5

    if btype == "tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2.0 * (_value + 1))
    elif btype == "sigmoid":
        sigmoid_scale = 2.0
        _value = sigmoid(sigmoid_scale * k * (10 * z - t))
        output = round(upperbound_timestep * (_value))
    else:
        raise NotImplementedError("btype is incorrect")
    return int(output)


def projection_matrix(E: torch.Tensor) -> torch.Tensor:
    """Calculate projection matrix onto subspace spanned by columns of E."""
    orig_dtype = E.dtype
    E32 = E.float()
    gram = E32.T @ E32
    eps = 1e-6
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram_reg = gram + eps * eye
    P32 = E32 @ torch.pinverse(gram_reg) @ E32.T
    return P32.to(orig_dtype)


def projection_and_orthogonal(
    input_embeddings: torch.Tensor,
    masked_input_subspace_projection: torch.Tensor,
    concept_subspace_projection: torch.Tensor,
) -> torch.Tensor:
    """Create orthogonal projection text embeddings."""
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection

    device = ie.device
    out_dtype = ie.dtype

    ms32 = ms.float()
    cs32 = cs.float()

    uncond_e, text_e = ie.chunk(2)
    text_e_s = text_e.squeeze(0)
    L, D = text_e_s.shape

    I32 = torch.eye(D, device=device, dtype=torch.float32)
    text_e_s32 = text_e_s.float()
    new_text_e32 = (I32 - cs32) @ ms32 @ text_e_s32.T
    new_text_e32 = new_text_e32.T
    new_text_e = new_text_e32.to(out_dtype)

    new_embeddings = torch.cat([uncond_e, new_text_e.unsqueeze(0)], dim=0)
    return new_embeddings


def safree_projection(
    input_embeddings: torch.Tensor,
    p_emb: torch.Tensor,
    masked_input_subspace_projection: torch.Tensor,
    concept_subspace_projection: torch.Tensor,
    alpha: float = 0.0,
    max_length: int = 77,
    logger=None,
) -> torch.Tensor:
    """Token-wise distance-based safe token preservation / trigger token replacement."""
    ie = input_embeddings
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection

    device = ie.device
    out_dtype = ie.dtype

    n_t, D = p_emb.shape

    p32 = p_emb.float()
    ms32 = ms.float()
    cs32 = cs.float()

    I32 = torch.eye(D, device=device, dtype=torch.float32)
    I_m_cs32 = I32 - cs32

    dist_vec = I_m_cs32 @ p32.T
    dist_p_emb = torch.norm(dist_vec, dim=0)

    if n_t > 1:
        sum_all = dist_p_emb.sum()
        mean_dist = (sum_all - dist_p_emb) / (n_t - 1)
    else:
        mean_dist = dist_p_emb.clone()

    rm_vector = (dist_p_emb < (1.0 + alpha) * mean_dist)
    n_removed = int(n_t - rm_vector.sum().item())
    if logger is not None:
        logger.log(f"Among {n_t} tokens, we remove {n_removed}.")
    else:
        print(f"[SAFREE] Among {n_t} tokens, we remove {n_removed}.")

    mask_bool = torch.zeros(max_length, device=device, dtype=torch.bool)
    mask_bool[1 : n_t + 1] = rm_vector

    uncond_e, text_e = ie.chunk(2)
    text_e_s = text_e.squeeze(0)
    L, D = text_e_s.shape

    text_e_s32 = text_e_s.float()
    new_text_e32 = I_m_cs32 @ ms32 @ text_e_s32.T
    new_text_e = new_text_e32.T.to(out_dtype)

    mask_2d = mask_bool.unsqueeze(1).expand_as(text_e_s)
    merged_text_e = torch.where(mask_2d, text_e_s, new_text_e).to(out_dtype)

    new_embeddings = torch.cat([uncond_e, merged_text_e.unsqueeze(0)], dim=0)
    return new_embeddings


# =========================
# Nudity 4-Class Configuration
# =========================
NUDITY_4CLASS_CONFIG = {
    "benign": 0,
    "safe_clothed": 1,
    "harm_nude": 2,
    "harm_color": 3,
    "safe_classes": [0, 1],
    "harm_classes": [2, 3],
    "guidance_target_safe": 1,
    "class_names": {
        0: "benign",
        1: "safe_clothed",
        2: "harm_nude",
        3: "harm_color"
    }
}


# =========================
# Negative Prompt Space
# =========================
def get_negative_prompt_space(category: str) -> List[str]:
    """Get negative prompt space for unsafe concept removal."""
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


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
    """Schedules spatial threshold across denoising steps."""

    def __init__(
        self,
        strategy: str = "linear_decrease",
        start_value: float = 0.7,
        end_value: float = 0.3,
        total_steps: int = 50
    ):
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step: int) -> float:
        if self.strategy == "constant":
            return self.start_value

        t = current_step / max(self.total_steps - 1, 1)

        if self.strategy == "linear_decrease":
            return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "linear_increase":
            return self.start_value + (self.end_value - self.start_value) * t
        elif self.strategy == "cosine_anneal":
            return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        else:
            return self.start_value


# =========================
# Harm Class Detector
# =========================
class HarmClassDetector:
    """Detects if current latent shows harmful content."""

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_4CLASS_CONFIG,
        device: str = "cuda"
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        self.safe_classes = config["safe_classes"]
        self.harm_classes = config["harm_classes"]
        self.target_safe = config["guidance_target_safe"]
        self.class_names = config["class_names"]

    def detect_harm(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[bool, Optional[int], int, Dict]:
        """Detect if current latent shows harmful content."""
        with torch.no_grad():
            latent_input = latent.to(dtype=self.classifier_dtype)

            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            norm_timestep = timestep.float() / 1000.0
            logits = self.classifier(latent_input, norm_timestep)
            max_class = logits.argmax(dim=1)[0].item()
            is_harmful = max_class in self.harm_classes
            harm_class = max_class if is_harmful else None
            safe_class = self.target_safe

            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "all_probs": probs,
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "max_logit": logits[0, max_class].item(),
                "max_prob": probs[max_class],
                "is_harmful": is_harmful,
                "harm_class": harm_class,
                "harm_class_name": self.class_names[harm_class] if harm_class is not None else None,
                "safe_class": safe_class,
                "safe_class_name": self.class_names[safe_class]
            }

        return is_harmful, harm_class, safe_class, info


# =========================
# GradCAM Stats Loader
# =========================
def load_gradcam_stats_map(stats_dir: str) -> Dict[int, Dict[str, float]]:
    """Load per-class GradCAM statistics."""
    stats_dir = Path(stats_dir)
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }

    stats_map: Dict[int, Dict[str, float]] = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if not path.exists():
            print(f"[Warning] GradCAM stats file not found: {path}")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}

    return stats_map


# =========================
# Selective Spatial Mask Generator
# =========================
class SelectiveSpatialMaskGenerator:
    """Generates spatial masks when harmful content detected."""

    def __init__(
        self,
        classifier_model,
        harm_detector: HarmClassDetector,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats_map: Optional[Dict[int, Dict[str, float]]] = None
    ):
        self.classifier = classifier_model
        self.harm_detector = harm_detector
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.gradcam_stats_map = gradcam_stats_map

        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )

        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        self.stats = {
            'total_steps': 0,
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': [],
            'harm_class_history': []
        }

    def _apply_cdf_normalization(self, heatmap: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        normal = Normal(
            torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype),
            torch.tensor(1.0, device=heatmap.device, dtype=heatmap.dtype)
        )
        return normal.cdf(z)

    def generate_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], int, Dict]:
        """Generate spatial mask if harmful content detected."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        is_harmful, harm_class, safe_class, detection_info = self.harm_detector.detect_harm(
            latent=latent,
            timestep=timestep
        )

        self.stats['total_steps'] += 1
        if is_harmful:
            self.stats['harmful_steps'] += 1

        if not is_harmful:
            self.stats['step_history'].append({
                'step': current_step,
                'spatial_threshold': spatial_threshold,
                'is_harmful': False,
                'max_class': detection_info['max_class'],
                'max_class_name': detection_info['max_class_name'],
                'mask_ratio': 0.0
            })
            self.stats['harm_class_history'].append(None)

            if self.debug:
                print(f"  [Step {current_step}] Safe ({detection_info['max_class_name']}) - Skipping CG")

            return False, None, None, None, safe_class, detection_info

        self.stats['guidance_applied'] += 1

        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        use_abs = (self.gradcam_stats_map is not None) and (harm_class in self.gradcam_stats_map)
        gradcam_normalize_flag = not use_abs

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=norm_timestep,
                target_class=harm_class,
                normalize=gradcam_normalize_flag
            )

        if use_abs:
            stats = self.gradcam_stats_map[harm_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])

        mask = (heatmap >= spatial_threshold).float()
        mask_ratio = mask.mean().item()

        self.stats['step_history'].append({
            'step': current_step,
            'spatial_threshold': spatial_threshold,
            'is_harmful': True,
            'harm_class': harm_class,
            'harm_class_name': detection_info['harm_class_name'],
            'mask_ratio': mask_ratio
        })
        self.stats['harm_class_history'].append(harm_class)

        if self.debug:
            print(f"  [Step {current_step}] Harmful ({detection_info['harm_class_name']}) - "
                  f"thr={spatial_threshold:.3f}, mask={mask_ratio:.1%}")

        return True, mask, heatmap if return_heatmap else None, harm_class, safe_class, detection_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': [],
            'harm_class_history': []
        }


# =========================
# Selective Spatial Guidance
# =========================
class SelectiveSpatialGuidance:
    """Applies bidirectional classifier guidance to harmful regions."""

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_4CLASS_CONFIG,
        device: str = "cuda",
        use_bidirectional: bool = True
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.use_bidirectional = use_bidirectional
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def compute_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        harm_class: int,
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """Compute spatially-masked bidirectional gradient."""
        with torch.enable_grad():
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            norm_timestep = timestep.float() / 1000.0

            if self.use_bidirectional:
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, norm_timestep)
                safe_logit = logits_safe[:, safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, norm_timestep)
                harmful_logit = logits_harmful[:, harm_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                grad = grad_safe - harmful_scale * grad_harmful
            else:
                logits = self.classifier(latent_input, norm_timestep)
                safe_logit = logits[:, safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        mask_expanded = spatial_mask.unsqueeze(1)
        weight_map = mask_expanded * guidance_scale + (1 - mask_expanded) * base_guidance_scale
        weighted_grad = grad * weight_map
        weighted_grad = weighted_grad.to(dtype=latent.dtype)

        return weighted_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        harm_class: int,
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        weighted_grad = self.compute_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            harm_class=harm_class,
            safe_class=safe_class,
            guidance_scale=guidance_scale,
            harmful_scale=harmful_scale,
            base_guidance_scale=base_guidance_scale
        )
        return latent + weighted_grad


# =========================
# SAFREE++ Spatial CG Pipeline
# =========================
class SAFREESpatialCGPipeline:
    """
    Combined SAFREE + Spatial CG Pipeline

    Applies both:
    1. SAFREE text-level projection (removes unsafe tokens)
    2. Spatial CG image-level guidance (targets harmful regions)
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        classifier_model,
        mask_generator: SelectiveSpatialMaskGenerator,
        guidance_module: SelectiveSpatialGuidance,
        threshold_scheduler: AdaptiveSpatialThresholdScheduler,
        device: str = "cuda",
        debug: bool = False
    ):
        self.pipe = pipe
        self.classifier = classifier_model
        self.mask_generator = mask_generator
        self.guidance_module = guidance_module
        self.threshold_scheduler = threshold_scheduler
        self.device = device
        self.debug = debug

        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    def _build_causal_attention_mask(self, bsz, seq_len, dtype, device):
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def _encode_negative_prompt_space(
        self, negative_prompt_space, max_length, num_images_per_prompt
    ):
        """Encode negative prompt space for SAFREE projection."""
        uncond_input = self.tokenizer(
            negative_prompt_space,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=uncond_input.attention_mask.to(self.device),
        )
        return uncond_embeddings.pooler_output

    def _masked_encode_prompt(self, prompt: Union[str, List[str]]):
        """Encode prompt with leave-one-out masking for SAFREE."""
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        n_real_tokens = untruncated_ids.shape[1] - 2

        if untruncated_ids.shape[1] > self.tokenizer.model_max_length:
            untruncated_ids = untruncated_ids[:, : self.tokenizer.model_max_length]
            n_real_tokens = self.tokenizer.model_max_length - 2

        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)
        for i in range(n_real_tokens):
            masked_ids[i, i + 1] = 0

        masked_embeddings = self.text_encoder(masked_ids.to(self.device), attention_mask=None)
        return masked_embeddings.pooler_output

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        """Encode text prompt with CFG support."""
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.to(self.device) if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask else None

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask
        )[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens, padding="max_length", max_length=max_length,
                truncation=True, return_tensors="pt"
            )

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=None
            )[0]

            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        return text_embeddings, text_input_ids, text_inputs.attention_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        # SAFREE parameters
        safree_enabled: bool = True,
        safree_alpha: float = 0.01,
        svf_enabled: bool = False,
        svf_up_t: int = 10,
        category: str = "nudity",
        # Spatial CG parameters
        spatial_cg_enabled: bool = True,
        cg_guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0,
        skip_if_safe: bool = False,
        guidance_start_step: int = 0,
        guidance_end_step: int = 50,
        # Output
        output_type: str = "pil",
    ):
        """
        Generate images with SAFREE text projection + Spatial CG guidance.
        """
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # Get negative prompt space for SAFREE
        negative_prompt_space = get_negative_prompt_space(category)
        if negative_prompt is None:
            negative_prompt = ", ".join(negative_prompt_space)

        # Encode prompt
        text_embeddings, text_input_ids, attention_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # SAFREE: Compute projection matrices
        if safree_enabled and negative_prompt_space:
            negspace_text_embeddings = self._encode_negative_prompt_space(
                negative_prompt_space,
                max_length=self.tokenizer.model_max_length,
                num_images_per_prompt=num_images_per_prompt
            )
            project_matrix = projection_matrix(negspace_text_embeddings.T)

            masked_embs = self._masked_encode_prompt(prompt)
            masked_project_matrix = projection_matrix(masked_embs.T)

            # Apply SAFREE projection
            rescaled_text_embeddings = safree_projection(
                text_embeddings,
                masked_embs,
                masked_project_matrix,
                project_matrix,
                alpha=safree_alpha,
                max_length=self.tokenizer.model_max_length,
            )

            # Self-Validation Filter
            if svf_enabled:
                proj_ort = projection_and_orthogonal(
                    text_embeddings, masked_project_matrix, project_matrix
                )

                _, text_e = text_embeddings.chunk(2)
                s_attn_mask = (attention_mask.squeeze() == 1) if attention_mask is not None else torch.ones(
                    text_e.shape[1], dtype=torch.bool, device=text_e.device
                )

                text_e = text_e.squeeze(0)
                _, proj_ort_e = proj_ort.chunk(2)
                proj_ort_e = proj_ort_e.squeeze(0)

                proj_ort_e_act = proj_ort_e[s_attn_mask]
                text_e_act = text_e[s_attn_mask]

                # Check if there are active tokens
                if proj_ort_e_act.numel() == 0 or text_e_act.numel() == 0:
                    print(f"[SAFREE SVF] No active tokens, skipping SVF adjustment")
                    beta_adjusted = -1
                else:
                    sim_org_onp_act = F.cosine_similarity(proj_ort_e_act.float(), text_e_act.float(), dim=-1)
                    beta = (1 - sim_org_onp_act.mean().item())

                    # Handle NaN case (can happen with adversarial prompts)
                    if math.isnan(beta):
                        print(f"[SAFREE SVF] Warning: beta is NaN, skipping SVF adjustment")
                        beta_adjusted = -1
                    else:
                        beta_adjusted = f_beta(beta, upperbound_timestep=svf_up_t, concept_type=category)
                        print(f"[SAFREE SVF] beta={beta:.4f}, adjusted_beta={beta_adjusted}")
            else:
                beta_adjusted = -1
        else:
            rescaled_text_embeddings = text_embeddings
            beta_adjusted = -1

        # Prepare latents
        latents_shape = (
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height // 8,
            width // 8,
        )
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=text_embeddings.dtype)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Scale initial noise
        latents = latents * self.scheduler.init_noise_sigma

        # Reset spatial CG statistics
        if spatial_cg_enabled:
            self.mask_generator.reset_statistics()

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="SAFREE++ Spatial CG")):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Choose embeddings (SAFREE or original)
            if svf_enabled:
                use_rescaled = safree_enabled and (i <= beta_adjusted)
            else:
                use_rescaled = safree_enabled

            _text_embeddings = rescaled_text_embeddings if use_rescaled else text_embeddings

            # UNet forward
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=_text_embeddings
            ).sample

            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Spatial CG: Apply classifier guidance after scheduler step
            if spatial_cg_enabled and guidance_start_step <= i <= guidance_end_step:
                spatial_threshold = self.threshold_scheduler.get_threshold(i)

                should_guide, spatial_mask, _, harm_class, safe_class, detection_info = \
                    self.mask_generator.generate_mask(
                        latent=latents,
                        timestep=t,
                        spatial_threshold=spatial_threshold,
                        current_step=i
                    )

                if should_guide and spatial_mask is not None:
                    latents = self.guidance_module.apply_guidance(
                        latent=latents,
                        timestep=t,
                        spatial_mask=spatial_mask,
                        harm_class=harm_class,
                        safe_class=safe_class,
                        guidance_scale=cg_guidance_scale,
                        harmful_scale=harmful_scale,
                        base_guidance_scale=base_guidance_scale
                    )

        # Decode latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

        # Print statistics
        if spatial_cg_enabled:
            stats = self.mask_generator.get_statistics()
            print(f"\n[Spatial CG Stats] Total: {stats['total_steps']}, "
                  f"Harmful: {stats['harmful_steps']}, "
                  f"Guided: {stats['guidance_applied']}")

        return image


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="SAFREE++ Spatial Classifier Guidance")

    # Model & Generation
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="./results/safree_spatial_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # SAFREE parameters
    parser.add_argument("--safree", action="store_true", help="Enable SAFREE text projection")
    parser.add_argument("--safree_alpha", type=float, default=0.01, help="SAFREE alpha parameter")
    parser.add_argument("--svf", action="store_true", help="Enable Self-Validation Filter")
    parser.add_argument("--svf_up_t", type=int, default=10, help="SVF upperbound timestep")
    parser.add_argument("--category", type=str, default="nudity",
                        choices=["nudity", "sexual", "violence"],
                        help="Category for negative prompt space")

    # Classifier (Spatial CG)
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Nudity 4-class classifier checkpoint path")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None,
                        help="Directory containing per-class GradCAM statistics")

    # Spatial CG parameters
    parser.add_argument("--spatial_cg", action="store_true", help="Enable Spatial CG")
    parser.add_argument("--cg_guidance_scale", type=float, default=5.0,
                        help="Classifier guidance strength")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7,
                        help="Initial spatial threshold")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="Final spatial threshold")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="Spatial threshold scheduling strategy")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale")
    parser.add_argument("--base_guidance_scale", type=float, default=0.0,
                        help="Base guidance scale for non-harmful regions")
    parser.add_argument("--skip_if_safe", action="store_true",
                        help="Skip CG entirely if classifier detects safe/benign")

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image(image, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(filepath)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device

    print("\n" + "="*80)
    print("SAFREE++ SPATIAL CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"SAFREE enabled: {args.safree}")
    print(f"Spatial CG enabled: {args.spatial_cg}")
    print("="*80 + "\n")

    # Load prompts
    print("[1/5] Loading prompts...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load pipeline
    print("\n[2/5] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("  Pipeline loaded")

    # Load classifier
    print(f"\n[3/5] Loading 4-class classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=4
    ).to(device)
    classifier.eval()
    print("  Classifier loaded")

    # Load GradCAM stats
    gradcam_stats_map = None
    if args.gradcam_stats_dir:
        print(f"\n[4/5] Loading GradCAM statistics from {args.gradcam_stats_dir}...")
        gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir)
    else:
        print("\n[4/5] No GradCAM stats provided -> per-image normalization")

    # Initialize components
    print("\n[5/5] Initializing components...")

    harm_detector = HarmClassDetector(
        classifier_model=classifier,
        config=NUDITY_4CLASS_CONFIG,
        device=device
    )

    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )

    mask_generator = SelectiveSpatialMaskGenerator(
        classifier_model=classifier,
        harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats_map=gradcam_stats_map
    )

    guidance_module = SelectiveSpatialGuidance(
        classifier_model=classifier,
        config=NUDITY_4CLASS_CONFIG,
        device=device,
        use_bidirectional=True
    )

    # Create combined pipeline
    safree_spatial_cg = SAFREESpatialCGPipeline(
        pipe=pipe,
        classifier_model=classifier,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        device=device,
        debug=args.debug
    )

    print("  All components initialized")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    print("\n" + "="*80)
    print("GENERATION START")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"SAFREE: {args.safree} (alpha={args.safree_alpha})")
    print(f"Spatial CG: {args.spatial_cg} (scale={args.cg_guidance_scale})")
    print(f"Threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print("="*80 + "\n")

    generator = torch.Generator(device=device)

    total_images = 0
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            generator.manual_seed(seed)

            images = safree_spatial_cg(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                num_images_per_prompt=1,
                generator=generator,
                # SAFREE
                safree_enabled=args.safree,
                safree_alpha=args.safree_alpha,
                svf_enabled=args.svf,
                svf_up_t=args.svf_up_t,
                category=args.category,
                # Spatial CG
                spatial_cg_enabled=args.spatial_cg,
                cg_guidance_scale=args.cg_guidance_scale,
                harmful_scale=args.harmful_scale,
                base_guidance_scale=args.base_guidance_scale,
                skip_if_safe=args.skip_if_safe,
                guidance_start_step=args.guidance_start_step,
                guidance_end_step=args.guidance_end_step,
            )

            # Save image
            safe_prompt = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in prompt)
            safe_prompt = safe_prompt[:50].strip().replace(' ', '_')
            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_path = output_dir / filename
            save_image(images[0], save_path)
            print(f"  Saved: {save_path}")

            total_images += 1

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total images generated: {total_images}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
