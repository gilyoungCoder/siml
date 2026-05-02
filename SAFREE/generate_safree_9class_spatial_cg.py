#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE++ 9-Class Spatial Classifier Guidance (Generic for all I2P concepts)

Combines two complementary safety mechanisms:
1. SAFREE (Text-level): Projects text embeddings to remove unsafe concept subspace
2. Spatial CG (Image-level): Applies classifier guidance to harmful regions

9-Class Structure:
  - Class 0: benign
  - Classes 1,3,5,7: harm classes (harm0, harm1, harm2, harm3)
  - Classes 2,4,6,8: safe classes (safe0, safe1, safe2, safe3)

Harm-Safe Pairs:
  - harm0(1) -> safe0(2)
  - harm1(3) -> safe1(4)
  - harm2(5) -> safe2(6)
  - harm3(7) -> safe3(8)

Supported Concepts:
  - harassment, hate, illegal, selfharm, shocking, violence (excluding sexual)
"""

import os
import sys
import json
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "SoftDelete+CG"))

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# =========================
# SAFREE Helper Functions
# =========================
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def f_beta(z: float, btype: str = "sigmoid", upperbound_timestep: int = 10, concept_type: str = "nudity") -> int:
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
# 9-Class Configuration
# =========================
I2P_9CLASS_CONFIG = {
    "benign": 0,
    "harm_classes": [1, 3, 5, 7],
    "safe_classes": [2, 4, 6, 8],
    "harm_to_safe": {1: 2, 3: 4, 5: 6, 7: 8},
    "class_names": {
        0: "benign",
        1: "harm0", 2: "safe0",
        3: "harm1", 4: "safe1",
        5: "harm2", 6: "safe2",
        7: "harm3", 8: "safe3"
    }
}


# =========================
# Negative Prompt Space per Concept
# =========================
CONCEPT_NEGATIVE_PROMPTS = {
    "harassment": [
        "harassment", "bullying", "intimidation", "threatening", "stalking",
        "verbal abuse", "humiliation", "mockery", "ridicule", "insult",
        "degradation", "taunting", "trolling", "cyberbullying", "hate speech"
    ],
    "hate": [
        "hate", "hatred", "discrimination", "racism", "sexism", "bigotry",
        "prejudice", "intolerance", "xenophobia", "homophobia", "antisemitism",
        "islamophobia", "supremacy", "extremism", "radicalism"
    ],
    "illegal": [
        "illegal activity", "crime", "criminal", "theft", "robbery", "burglary",
        "fraud", "forgery", "smuggling", "trafficking", "drug dealing",
        "money laundering", "hacking", "piracy", "counterfeiting"
    ],
    "selfharm": [
        "self-harm", "self-injury", "cutting", "suicide", "suicidal",
        "overdose", "self-mutilation", "depression", "eating disorder",
        "anorexia", "bulimia", "self-destruction", "harmful behavior"
    ],
    "shocking": [
        "shocking content", "disturbing", "graphic", "gore", "mutilation",
        "torture", "extreme violence", "death", "corpse", "dismemberment",
        "trauma", "horrific", "gruesome", "macabre", "grotesque"
    ],
    "violence": [
        "violence", "assault", "attack", "fight", "beating", "punching",
        "kicking", "hitting", "stabbing", "knife attack", "shooting", "gun violence",
        "aiming a gun", "strangling", "choking", "execution", "blood", "bloody scene",
        "open wound", "corpse", "explosion", "grenade blast", "bomb blast",
        "molotov cocktail", "arson"
    ],
    "nudity": [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ]
}


def get_negative_prompt_space(category: str) -> List[str]:
    return CONCEPT_NEGATIVE_PROMPTS.get(category, [])


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
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
# Dynamic Harm Class Detector (9-class)
# =========================
class DynamicHarmClassDetector:
    """Detects which harm class to guide from based on classifier output."""

    def __init__(
        self,
        classifier_model,
        config: Dict = I2P_9CLASS_CONFIG,
        device: str = "cuda",
        skip_if_safe: bool = True
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.skip_if_safe = skip_if_safe
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        self.harm_classes = config["harm_classes"]
        self.safe_classes = config["safe_classes"]
        self.harm_to_safe = config["harm_to_safe"]
        self.class_names = config["class_names"]
        self.benign_class = config["benign"]

    def detect_harm(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[bool, Optional[int], int, Dict]:
        """Detect which harm class is most active."""
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

            # Find max class
            max_class = logits.argmax(dim=1)[0].item()

            # Check if harmful
            is_harmful = max_class in self.harm_classes

            if is_harmful:
                harm_class = max_class
                safe_class = self.harm_to_safe[harm_class]
            elif self.skip_if_safe:
                harm_class = None
                safe_class = None
            else:
                # Default to highest harm logit
                harm_logits = logits[0, self.harm_classes]
                best_harm_idx = harm_logits.argmax().item()
                harm_class = self.harm_classes[best_harm_idx]
                safe_class = self.harm_to_safe[harm_class]

            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "all_probs": probs,
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "is_harmful": is_harmful,
                "harm_class": harm_class,
                "safe_class": safe_class
            }

        return is_harmful, harm_class, safe_class, info


# =========================
# GradCAM Stats Loader (9-class)
# =========================
def load_gradcam_stats_map(stats_dir: str, harm_classes: List[int] = [1, 3, 5, 7]) -> Dict[int, Dict[str, float]]:
    """Load per-class GradCAM statistics."""
    stats_dir = Path(stats_dir)
    stats_map: Dict[int, Dict[str, float]] = {}

    for cls in harm_classes:
        # Try different naming conventions
        possible_names = [
            f"gradcam_stats_class{cls}.json",
            f"gradcam_stats_harm{cls}.json",
            f"gradcam_stats_harm_class{cls}.json",
        ]

        for fname in possible_names:
            path = stats_dir / fname
            if path.exists():
                with open(path, "r") as f:
                    d = json.load(f)
                stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
                print(f"  Loaded stats for class {cls}: mean={d['mean']:.4f}, std={d['std']:.4f}")
                break

    return stats_map


# =========================
# Selective Spatial Mask Generator (9-class)
# =========================
class SelectiveSpatialMaskGenerator:
    def __init__(
        self,
        classifier_model,
        harm_detector: DynamicHarmClassDetector,
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
        current_step: Optional[int] = None
    ) -> Tuple[bool, Optional[torch.Tensor], Optional[int], Optional[int], Dict]:
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

        if not is_harmful or harm_class is None:
            self.stats['step_history'].append({
                'step': current_step,
                'is_harmful': False,
                'max_class': detection_info['max_class'],
                'mask_ratio': 0.0
            })
            self.stats['harm_class_history'].append(None)

            if self.debug:
                print(f"  [Step {current_step}] Safe ({detection_info['max_class_name']}) - Skipping CG")

            return False, None, None, None, detection_info

        self.stats['guidance_applied'] += 1

        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        use_abs = (self.gradcam_stats_map is not None) and (harm_class in self.gradcam_stats_map)
        gradcam_normalize_flag = not use_abs

        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(
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
            'is_harmful': True,
            'harm_class': harm_class,
            'safe_class': safe_class,
            'mask_ratio': mask_ratio
        })
        self.stats['harm_class_history'].append(harm_class)

        if self.debug:
            print(f"  [Step {current_step}] Harmful (class {harm_class}) -> Safe (class {safe_class}) - "
                  f"thr={spatial_threshold:.3f}, mask={mask_ratio:.1%}")

        return True, mask, harm_class, safe_class, detection_info

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
# Selective Spatial Guidance (9-class)
# =========================
class SelectiveSpatialGuidance:
    def __init__(
        self,
        classifier_model,
        device: str = "cuda",
        use_bidirectional: bool = True
    ):
        self.classifier = classifier_model
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
# SAFREE++ 9-Class Spatial CG Pipeline
# =========================
class SAFREE9ClassSpatialCGPipeline:
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

    def _encode_negative_prompt_space(self, negative_prompt_space, max_length, num_images_per_prompt):
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

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
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
        category: str = "violence",
        # Spatial CG parameters
        spatial_cg_enabled: bool = True,
        cg_guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0,
        guidance_start_step: int = 0,
        guidance_end_step: int = 50,
        output_type: str = "pil",
    ):
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        negative_prompt_space = get_negative_prompt_space(category)
        if negative_prompt is None and negative_prompt_space:
            negative_prompt = ", ".join(negative_prompt_space)

        text_embeddings, text_input_ids, attention_mask = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # SAFREE projection
        if safree_enabled and negative_prompt_space:
            negspace_text_embeddings = self._encode_negative_prompt_space(
                negative_prompt_space,
                max_length=self.tokenizer.model_max_length,
                num_images_per_prompt=num_images_per_prompt
            )
            project_matrix = projection_matrix(negspace_text_embeddings.T)

            masked_embs = self._masked_encode_prompt(prompt)
            masked_project_matrix = projection_matrix(masked_embs.T)

            rescaled_text_embeddings = safree_projection(
                text_embeddings,
                masked_embs,
                masked_project_matrix,
                project_matrix,
                alpha=safree_alpha,
                max_length=self.tokenizer.model_max_length,
            )

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

                    # Check for NaN
                    if math.isnan(beta):
                        print(f"[SAFREE SVF] beta is NaN, skipping SVF adjustment")
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

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma

        if spatial_cg_enabled:
            self.mask_generator.reset_statistics()

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc=f"SAFREE++ 9-Class ({category})")):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if svf_enabled:
                use_rescaled = safree_enabled and (i <= beta_adjusted)
            else:
                use_rescaled = safree_enabled

            _text_embeddings = rescaled_text_embeddings if use_rescaled else text_embeddings

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=_text_embeddings
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Spatial CG
            if spatial_cg_enabled and guidance_start_step <= i <= guidance_end_step:
                spatial_threshold = self.threshold_scheduler.get_threshold(i)

                should_guide, spatial_mask, harm_class, safe_class, _ = \
                    self.mask_generator.generate_mask(
                        latent=latents,
                        timestep=t,
                        spatial_threshold=spatial_threshold,
                        current_step=i
                    )

                if should_guide and spatial_mask is not None and harm_class is not None:
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

        # Decode
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.pipe.numpy_to_pil(image)

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
    parser = ArgumentParser(description="SAFREE++ 9-Class Spatial CG")

    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/safree_9class_spatial_cg")

    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # SAFREE
    parser.add_argument("--safree", action="store_true")
    parser.add_argument("--safree_alpha", type=float, default=0.01)
    parser.add_argument("--svf", action="store_true")
    parser.add_argument("--svf_up_t", type=int, default=10)
    parser.add_argument("--category", type=str, default="violence",
                        choices=["harassment", "hate", "illegal", "selfharm", "shocking", "violence", "nudity"])

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None)

    # Spatial CG
    parser.add_argument("--spatial_cg", action="store_true")
    parser.add_argument("--cg_guidance_scale", type=float, default=5.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3)
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease")
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--base_guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)
    parser.add_argument("--skip_if_safe", action="store_true", default=True)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("\n" + "="*80)
    print(f"SAFREE++ 9-CLASS SPATIAL CG ({args.category.upper()})")
    print("="*80)

    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=9
    ).to(device)
    classifier.eval()

    gradcam_stats_map = None
    if args.gradcam_stats_dir:
        gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir)

    harm_detector = DynamicHarmClassDetector(
        classifier_model=classifier,
        config=I2P_9CLASS_CONFIG,
        device=device,
        skip_if_safe=args.skip_if_safe
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
        device=device,
        use_bidirectional=True
    )

    pipeline = SAFREE9ClassSpatialCGPipeline(
        pipe=pipe,
        classifier_model=classifier,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        device=device,
        debug=args.debug
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device)

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            generator.manual_seed(seed)

            images = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                safree_enabled=args.safree,
                safree_alpha=args.safree_alpha,
                svf_enabled=args.svf,
                svf_up_t=args.svf_up_t,
                category=args.category,
                spatial_cg_enabled=args.spatial_cg,
                cg_guidance_scale=args.cg_guidance_scale,
                harmful_scale=args.harmful_scale,
                base_guidance_scale=args.base_guidance_scale,
                guidance_start_step=args.guidance_start_step,
                guidance_end_step=args.guidance_end_step,
            )

            safe_prompt = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in prompt)[:50].strip().replace(' ', '_')
            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(images[0], output_dir / filename)
            print(f"  Saved: {output_dir / filename}")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
