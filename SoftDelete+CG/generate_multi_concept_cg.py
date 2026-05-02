#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Concept Selective Classifier Guidance for Machine Unlearning

Extends Selective CG to handle multiple harmful concepts simultaneously:
  - Nudity removal using nudity classifier
  - Violence removal using violence classifier

Key Innovation:
  - Each classifier monitors and guides independently
  - Guidance gradients are combined when both concepts detected
  - Spatial masking per concept (concept-specific heatmaps)
  - Adaptive thresholds and weight scheduling per concept

Architecture:
  1. Two independent SelectiveGuidanceMonitor instances
  2. Two independent SpatiallyMaskedGuidance modules
  3. Combined gradient application strategy:
     - If only nudity detected: apply nudity guidance
     - If only violence detected: apply violence guidance
     - If both detected: combine both guidances (additive or weighted)

Benefits:
  - Simultaneous multi-concept erasure
  - Concept-specific spatial targeting
  - Independent parameter tuning per concept
  - Efficient single-pass generation
"""

import os
import sys
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
from geo_utils.selective_guidance_utils import (
    SelectiveGuidanceMonitor,
    SpatiallyMaskedGuidance,
    WeightScheduler,
    visualize_selective_guidance
)


# =========================
# Multi-Concept Guidance Manager
# =========================
class MultiConceptGuidanceManager:
    """
    Manages multiple concept-specific guidance modules.

    Each concept has:
      - Independent monitor for detection
      - Independent spatially masked guidance
      - Independent parameters (threshold, guidance scale, etc.)
    """

    def __init__(
        self,
        concepts: Dict[str, Dict],
        device: str = "cuda"
    ):
        """
        Args:
            concepts: Dictionary of concept configurations
                Example:
                {
                    'nudity': {
                        'monitor': SelectiveGuidanceMonitor(...),
                        'guidance': SpatiallyMaskedGuidance(...),
                        'weight_scheduler': WeightScheduler(...),
                        'enabled': True
                    },
                    'violence': {
                        'monitor': SelectiveGuidanceMonitor(...),
                        'guidance': SpatiallyMaskedGuidance(...),
                        'weight_scheduler': WeightScheduler(...),
                        'enabled': True
                    }
                }
        """
        self.concepts = concepts
        self.device = device
        self.detection_history = {name: [] for name in concepts.keys()}

    def detect_harmful_concepts(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        current_step: int
    ) -> Dict[str, bool]:
        """
        Detect which concepts are harmful in the current latent.

        Returns:
            Dictionary mapping concept name to detection result
        """
        detections = {}

        for concept_name, concept_config in self.concepts.items():
            if not concept_config['enabled']:
                detections[concept_name] = False
                continue

            monitor = concept_config['monitor']
            is_harmful, harmful_score, _ = monitor.detect_harmful(
                latent, timestep, current_step=current_step
            )

            detections[concept_name] = is_harmful
            self.detection_history[concept_name].append({
                'step': current_step,
                'detected': is_harmful,
                'score': harmful_score
            })

        return detections

    def compute_combined_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        current_step: int,
        detections: Dict[str, bool],
        combination_strategy: str = "additive"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined guidance from all detected harmful concepts.

        Args:
            latent: Current latent tensor
            timestep: Current timestep
            current_step: Current denoising step
            detections: Dict of concept detection results
            combination_strategy: How to combine gradients
                - "additive": Simple sum (original)
                - "direction_based": Normalize to unit vectors, combine directions, then scale
                - "orthogonal_projection": Use orthogonal projection for conflict resolution
                - "max", "weighted": Other strategies

        Returns:
            combined_grad: Combined guidance gradient
            concept_grads: Dictionary of individual concept gradients
        """
        concept_grads = {}
        concept_grads_raw = {}  # Store raw gradients before scaling
        concept_scales = {}  # Store scales for later
        detected_concepts = []

        # Collect gradients from detected concepts
        for concept_name, is_harmful in detections.items():
            if not is_harmful:
                continue

            concept_config = self.concepts[concept_name]
            monitor = concept_config['monitor']
            guidance_module = concept_config['guidance']
            weight_scheduler = concept_config.get('weight_scheduler')
            guidance_scale = concept_config.get('guidance_scale', 7.0)
            harmful_scale = concept_config.get('harmful_scale', 1.0)

            # Get spatial mask from monitor
            should_apply, spatial_mask, info = monitor.should_apply_guidance(
                latent, timestep, current_step
            )

            if not should_apply or spatial_mask is None:
                continue

            # Get weight multiplier for this step
            if weight_scheduler is not None:
                weight_mult = weight_scheduler.get_weight(current_step)
            else:
                weight_mult = 1.0

            # Compute spatially masked gradient WITHOUT scale (to get raw direction)
            # We'll apply scale later in direction-based combination
            grad_raw = guidance_module.compute_masked_gradient(
                latent=latent,
                timestep=timestep,
                spatial_mask=spatial_mask,
                guidance_scale=1.0,  # Use scale=1.0 to get raw direction
                harmful_scale=harmful_scale,  # Use actual harmful_scale for bidirectional guidance
                current_step=current_step
            )

            # Compute effective scale: guidance_scale * harmful_scale * weight_mult
            # harmful_scale is already applied in compute_masked_gradient for bidirectional guidance
            # So we only multiply by guidance_scale and weight_mult here
            effective_scale = guidance_scale * weight_mult

            # Store raw gradient and scale separately
            concept_grads_raw[concept_name] = grad_raw
            concept_scales[concept_name] = effective_scale
            detected_concepts.append(concept_name)

        # Multi-concept scale adjustment: reduce scale when multiple concepts detected
        # This prevents over-guidance when combining multiple gradients
        num_detected = len(concept_grads_raw)
        if num_detected > 1:
            # Apply scaling factor: 1/sqrt(N) to balance combined magnitude
            multi_concept_factor = 1.0 / (num_detected ** 0.5)
            for concept_name in concept_scales:
                concept_scales[concept_name] *= multi_concept_factor
            print(f"[Multi-concept adjustment] {num_detected} concepts detected, "
                  f"applying scale factor: {multi_concept_factor:.3f}")

        # Store scaled versions for backward compatibility
        for concept_name in concept_grads_raw:
            concept_grads[concept_name] = concept_grads_raw[concept_name] * concept_scales[concept_name]

        # Combine gradients
        if len(concept_grads) == 0:
            # No harmful concepts detected
            combined_grad = torch.zeros_like(latent)
        elif len(concept_grads) == 1:
            # Single concept detected - apply effective scale
            concept_name = list(concept_grads_raw.keys())[0]
            grad_raw = concept_grads_raw[concept_name]
            effective_scale = concept_scales[concept_name]
            combined_grad = grad_raw * effective_scale
        else:
            # Multiple concepts detected - analyze conflict and combine
            if 'nudity' in concept_grads and 'violence' in concept_grads:
                g_nudity_raw = concept_grads_raw['nudity']
                g_violence_raw = concept_grads_raw['violence']

                # Flatten gradients for analysis
                g_nudity_flat = g_nudity_raw.flatten()
                g_violence_flat = g_violence_raw.flatten()

                # Compute norms
                norm_nudity = torch.norm(g_nudity_flat)
                norm_violence = torch.norm(g_violence_flat)

                # Compute cosine similarity
                if norm_nudity > 1e-8 and norm_violence > 1e-8:
                    dot_product = torch.dot(g_nudity_flat, g_violence_flat)
                    cosine_sim = dot_product / (norm_nudity * norm_violence)
                else:
                    dot_product = torch.tensor(0.0)
                    cosine_sim = torch.tensor(0.0)

                # Log conflict metrics
                if not hasattr(self, 'gradient_conflicts'):
                    self.gradient_conflicts = []

                self.gradient_conflicts.append({
                    'step': current_step,
                    'dot_product': dot_product.item(),
                    'cosine_similarity': cosine_sim.item(),
                    'norm_nudity': norm_nudity.item(),
                    'norm_violence': norm_violence.item(),
                    'is_conflict': cosine_sim.item() < 0,
                    'strategy': combination_strategy
                })

            # Direction-based combination strategies
            if combination_strategy == "direction_based":
                combined_grad = self._combine_by_direction(
                    concept_grads_raw, concept_scales, current_step
                )
            elif combination_strategy == "orthogonal_projection":
                combined_grad = self._combine_by_orthogonal_projection(
                    concept_grads_raw, concept_scales, current_step
                )
            elif combination_strategy == "additive":
                # Simple addition (original behavior)
                combined_grad = sum(concept_grads.values())
            elif combination_strategy == "max":
                # Take maximum magnitude per pixel
                grad_stack = torch.stack(list(concept_grads.values()), dim=0)
                max_abs = grad_stack.abs().sum(dim=[2, 3])  # [num_concepts, B, C]
                max_idx = torch.argmax(max_abs.sum(dim=2), dim=0)  # [B]
                combined_grad = grad_stack[max_idx, torch.arange(grad_stack.shape[1])]
            elif combination_strategy == "weighted":
                # Weight by detection confidence (equal weighting for now)
                combined_grad = sum(concept_grads.values()) / len(concept_grads)
            else:
                raise ValueError(f"Unknown combination strategy: {combination_strategy}")

        return combined_grad, concept_grads

    def _combine_by_direction(
        self,
        concept_grads_raw: Dict[str, torch.Tensor],
        concept_scales: Dict[str, float],
        current_step: int
    ) -> torch.Tensor:
        """
        Combine gradients based on normalized directions.

        Strategy:
        1. Normalize each gradient to unit vector (extract direction)
        2. Average directions
        3. Apply unified scale

        This treats all concepts equally regardless of their magnitude.
        """
        if len(concept_grads_raw) == 0:
            return torch.zeros_like(list(concept_grads_raw.values())[0])

        # Normalize each gradient to get unit direction vectors
        # Also track original norms for final scaling
        directions = []
        original_norms = []
        for concept_name, grad_raw in concept_grads_raw.items():
            grad_flat = grad_raw.flatten()
            norm = torch.norm(grad_flat)
            original_norms.append(norm)
            if norm > 1e-8:
                direction = grad_raw / norm
            else:
                direction = grad_raw  # Keep zero gradient as is
            directions.append(direction)

        # Average all directions
        avg_direction = sum(directions) / len(directions)

        # Normalize the averaged direction
        avg_flat = avg_direction.flatten()
        avg_norm = torch.norm(avg_flat)
        if avg_norm > 1e-8:
            common_direction = avg_direction / avg_norm
        else:
            common_direction = avg_direction

        # Use average of original gradient norms as scale
        avg_original_norm = sum(original_norms) / len(original_norms)
        combined_grad = common_direction * avg_original_norm

        return combined_grad

    def _combine_by_orthogonal_projection(
        self,
        concept_grads_raw: Dict[str, torch.Tensor],
        concept_scales: Dict[str, float],
        current_step: int
    ) -> torch.Tensor:
        """
        Combine gradients using Restricted Gradient (RG) optimization.

        Strategy (based on RG paper):
        1. Normalize both gradients to unit vectors
        2. Compute restricted gradients by removing conflicting components:
           δ*_f = ∇L_f - (∇L_f · ∇L_r / ||∇L_r||²) ∇L_r
           δ*_r = ∇L_r - (∇L_r · ∇L_f / ||∇L_f||²) ∇L_f
        3. Combine: δ*_f + δ*_r

        This removes gradient conflicts while preserving each gradient's unique direction.
        """
        if len(concept_grads_raw) == 0:
            return torch.zeros_like(list(concept_grads_raw.values())[0])

        if len(concept_grads_raw) == 1:
            # Single concept - use the already scaled version from concept_grads
            # (grad_raw * effective_scale was already computed in compute_combined_guidance)
            concept_name = list(concept_grads_raw.keys())[0]
            grad_raw = concept_grads_raw[concept_name]
            # Just return the raw gradient with its natural magnitude
            # The scale will be applied via the original norm
            grad_flat = grad_raw.flatten()
            norm = torch.norm(grad_flat)
            return grad_raw  # Use original gradient magnitude without extra scaling

        # Handle 2-concept case
        if 'nudity' in concept_grads_raw and 'violence' in concept_grads_raw:
            concept_a_name, concept_b_name = 'nudity', 'violence'
        elif 'nudity' in concept_grads_raw and 'vangogh' in concept_grads_raw:
            concept_a_name, concept_b_name = 'nudity', 'vangogh'
        elif 'violence' in concept_grads_raw and 'vangogh' in concept_grads_raw:
            concept_a_name, concept_b_name = 'violence', 'vangogh'
        else:
            # Fallback for other combinations
            return self._combine_by_direction(concept_grads_raw, concept_scales, current_step)

        if concept_a_name and concept_b_name:
            g_n = concept_grads_raw['nudity']  # [B, C, H, W]
            g_v = concept_grads_raw['violence']  # [B, C, H, W]
            scale_n = concept_scales['nudity']
            scale_v = concept_scales['violence']

            B, C, H, W = g_n.shape

            # Reshape to [B*C, H*W] for vectorized operations
            # Each column is a (B*C)-dimensional vector for one pixel
            g_n_flat = g_n.reshape(B * C, H * W)  # [BC, HW]
            g_v_flat = g_v.reshape(B * C, H * W)  # [BC, HW]

            # Compute norms for each pixel (vectorized)
            norms_n = torch.norm(g_n_flat, dim=0)  # [HW]
            norms_v = torch.norm(g_v_flat, dim=0)  # [HW]

            # Normalize to unit vectors (avoid division by zero)
            eps = 1e-8
            dir_n = g_n_flat / (norms_n.unsqueeze(0) + eps)  # [BC, HW]
            dir_v = g_v_flat / (norms_v.unsqueeze(0) + eps)  # [BC, HW]

            # Compute dot products for all pixels at once
            dot_products = (dir_n * dir_v).sum(dim=0)  # [HW]

            # Restricted Gradient Method (vectorized)
            # Project out the conflicting component, then scale back to original magnitude

            # proj_n_on_v = dot_product * dir_v for each pixel
            proj_n_on_v = dot_products.unsqueeze(0) * dir_v  # [BC, HW]
            restricted_n_dir = dir_n - proj_n_on_v  # [BC, HW] - unit vector

            proj_v_on_n = dot_products.unsqueeze(0) * dir_n  # [BC, HW]
            restricted_v_dir = dir_v - proj_v_on_n  # [BC, HW] - unit vector

            # Scale back to original magnitudes AND apply effective scales
            restricted_n = restricted_n_dir * norms_n.unsqueeze(0) * scale_n  # [BC, HW]
            restricted_v = restricted_v_dir * norms_v.unsqueeze(0) * scale_v  # [BC, HW]

            # Combine restricted gradients (with scales applied)
            combined_grad_both = restricted_n + restricted_v  # [BC, HW]

            # Handle fallback cases (where restricted gradients cancel out)
            combined_norms = torch.norm(combined_grad_both, dim=0)  # [HW]
            mask_cancel = combined_norms < eps  # [HW]
            if mask_cancel.any():
                # Fallback: use average of scaled original gradients
                avg_grad = (g_n_flat * scale_n + g_v_flat * scale_v) / 2
                combined_grad_both[:, mask_cancel] = avg_grad[:, mask_cancel]

            # Use the combined restricted gradients directly
            combined_flat = combined_grad_both  # [BC, HW]

            # Handle pixels where only one concept exists (apply scales)
            mask_only_n = (norms_n > eps) & (norms_v <= eps)
            mask_only_v = (norms_v > eps) & (norms_n <= eps)
            combined_flat[:, mask_only_n] = g_n_flat[:, mask_only_n] * scale_n
            combined_flat[:, mask_only_v] = g_v_flat[:, mask_only_v] * scale_v

            # Reshape back to [B, C, H, W]
            combined_grad = combined_flat.reshape(B, C, H, W)

            # Log statistics
            mask_both = (norms_n > eps) & (norms_v > eps)
            num_valid_pixels = mask_both.sum().item()
            if num_valid_pixels > 0:
                avg_norm_n = norms_n[mask_both].mean().item()
                avg_norm_v = norms_v[mask_both].mean().item()
                print(f"[Step {current_step}] Pixel-wise RG (vectorized): {num_valid_pixels}/{H*W} valid pixels, "
                      f"avg_norm_n={avg_norm_n:.4f}, avg_norm_v={avg_norm_v:.4f}")

            return combined_grad

        else:
            # Fallback for other combinations: use direction-based average
            return self._combine_by_direction(concept_grads_raw, concept_scales, current_step)

    def get_statistics(self) -> Dict:
        """Get detection statistics for all concepts."""
        stats = {}

        for concept_name, history in self.detection_history.items():
            if len(history) == 0:
                stats[concept_name] = {
                    'total_steps': 0,
                    'detected_steps': 0,
                    'detection_rate': 0.0
                }
            else:
                detected = sum(1 for h in history if h['detected'])
                stats[concept_name] = {
                    'total_steps': len(history),
                    'detected_steps': detected,
                    'detection_rate': detected / len(history)
                }

        return stats


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Multi-Concept Selective Classifier Guidance")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/multi_concept_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Nudity Classifier Parameters
    parser.add_argument("--nudity_classifier", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Nudity classifier checkpoint")
    parser.add_argument("--nudity_enabled", action="store_true",
                        help="Enable nudity concept erasure")
    parser.add_argument("--nudity_harmful_threshold", type=float, default=-0.5,
                        help="Harmful detection threshold for nudity")
    parser.add_argument("--nudity_spatial_threshold", type=float, default=0.3,
                        help="Spatial masking threshold for nudity")
    parser.add_argument("--nudity_guidance_scale", type=float, default=7.0,
                        help="Guidance scale for nudity")
    parser.add_argument("--nudity_harmful_scale", type=float, default=1.25,
                        help="Harmful repulsion scale for nudity")
    parser.add_argument("--nudity_harmful_class", type=int, default=2,
                        help="Harmful class index for nudity (2=Full)")
    parser.add_argument("--nudity_safe_class", type=int, default=1,
                        help="Safe class index for nudity (1=Safe)")

    # Violence Classifier Parameters
    parser.add_argument("--violence_classifier", type=str,
                        default="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth",
                        help="Violence classifier checkpoint")
    parser.add_argument("--violence_enabled", action="store_true",
                        help="Enable violence concept erasure")
    parser.add_argument("--violence_harmful_threshold", type=float, default=-0.5,
                        help="Harmful detection threshold for violence")
    parser.add_argument("--violence_spatial_threshold", type=float, default=0.3,
                        help="Spatial masking threshold for violence")
    parser.add_argument("--violence_guidance_scale", type=float, default=7.0,
                        help="Guidance scale for violence")
    parser.add_argument("--violence_harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale for violence")
    parser.add_argument("--violence_harmful_class", type=int, default=2,
                        help="Harmful class index for violence (2=Full)")
    parser.add_argument("--violence_safe_class", type=int, default=1,
                        help="Safe class index for violence (1=Safe)")

    # VanGogh Classifier Parameters
    parser.add_argument("--vangogh_classifier", type=str,
                        default="./work_dirs/vangogh_three_class_diff/checkpoint/step_11400/classifier.pth",
                        help="VanGogh classifier checkpoint")
    parser.add_argument("--vangogh_enabled", action="store_true",
                        help="Enable VanGogh style erasure")
    parser.add_argument("--vangogh_harmful_threshold", type=float, default=-0.5,
                        help="Harmful detection threshold for VanGogh")
    parser.add_argument("--vangogh_spatial_threshold", type=float, default=0.3,
                        help="Spatial masking threshold for VanGogh")
    parser.add_argument("--vangogh_guidance_scale", type=float, default=7.0,
                        help="Guidance scale for VanGogh")
    parser.add_argument("--vangogh_harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale for VanGogh")
    parser.add_argument("--vangogh_harmful_class", type=int, default=2,
                        help="Harmful class index for VanGogh (2=VanGogh style)")
    parser.add_argument("--vangogh_safe_class", type=int, default=1,
                        help="Safe class index for VanGogh (1=Safe)")

    # Common parameters
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance for both concepts")
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Adaptive threshold scheduling (shared)
    parser.add_argument("--use_adaptive_threshold", action="store_true",
                        help="Enable adaptive threshold scheduling")
    parser.add_argument("--threshold_strategy", type=str, default="cosine_anneal",
                        choices=["constant", "linear_increase", "linear_decrease", "cosine_anneal"],
                        help="Threshold scheduling strategy")
    parser.add_argument("--nudity_threshold_start", type=float, default=0.0,
                        help="Starting threshold for nudity")
    parser.add_argument("--nudity_threshold_end", type=float, default=-2.0,
                        help="Ending threshold for nudity")
    parser.add_argument("--violence_threshold_start", type=float, default=0.0,
                        help="Starting threshold for violence")
    parser.add_argument("--violence_threshold_end", type=float, default=-2.0,
                        help="Ending threshold for violence")
    parser.add_argument("--vangogh_threshold_start", type=float, default=0.0,
                        help="Starting threshold for VanGogh")
    parser.add_argument("--vangogh_threshold_end", type=float, default=-2.0,
                        help="Ending threshold for VanGogh")

    # Weight scheduling (shared)
    parser.add_argument("--use_weight_scheduling", action="store_true",
                        help="Enable guidance scale scheduling")
    parser.add_argument("--weight_strategy", type=str, default="cosine_anneal",
                        choices=["constant", "linear_increase", "linear_decrease",
                                 "cosine_anneal", "exponential_decay"],
                        help="Weight scheduling strategy")
    parser.add_argument("--nudity_weight_start", type=float, default=3.0,
                        help="Starting weight for nudity")
    parser.add_argument("--nudity_weight_end", type=float, default=0.5,
                        help="Ending weight for nudity")
    parser.add_argument("--violence_weight_start", type=float, default=3.0,
                        help="Starting weight for violence")
    parser.add_argument("--violence_weight_end", type=float, default=0.5,
                        help="Ending weight for violence")
    parser.add_argument("--vangogh_weight_start", type=float, default=3.0,
                        help="Starting weight for VanGogh")
    parser.add_argument("--vangogh_weight_end", type=float, default=0.5,
                        help="Ending weight for VanGogh")

    # Heatmap-weighted guidance (shared)
    parser.add_argument("--use_heatmap_weighted_guidance", action="store_true",
                        help="Use GradCAM heatmap values to weight guidance")

    # Gradient normalization (shared)
    parser.add_argument("--normalize_gradient", action="store_true",
                        help="Normalize gradients for stability")
    parser.add_argument("--gradient_norm_type", type=str, default="l2",
                        choices=["l2", "layer"],
                        help="Type of gradient normalization")

    # Multi-concept combination
    parser.add_argument("--combination_strategy", type=str, default="additive",
                        choices=["additive", "direction_based", "orthogonal_projection", "max", "weighted"],
                        help="Strategy for combining multiple concept gradients")

    # Grad-CAM layer
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save Grad-CAM and guidance visualizations")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from file."""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_image(image, filepath: Path):
    """Save PIL image to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(filepath)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Main Generation Function
# =========================
def generate_with_multi_concept_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    manager: MultiConceptGuidanceManager,
    args,
    output_dir: Path
):
    """Generate images with multi-concept classifier guidance."""
    device = pipe.device

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MULTI-CONCEPT SELECTIVE CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Enabled concepts:")
    if args.nudity_enabled:
        print(f"  ✓ Nudity (threshold={args.nudity_harmful_threshold}, scale={args.nudity_guidance_scale})")
    if args.violence_enabled:
        print(f"  ✓ Violence (threshold={args.violence_harmful_threshold}, scale={args.violence_guidance_scale})")
    if args.vangogh_enabled:
        print(f"  ✓ VanGogh (threshold={args.vangogh_harmful_threshold}, scale={args.vangogh_guidance_scale})")
    print(f"Combination strategy: {args.combination_strategy}")
    print("="*80 + "\n")

    # Generation loop
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] Prompt: {prompt}")

        for sample_idx in range(args.nsamples):
            # Reset detection history for new sample
            manager.detection_history = {name: [] for name in manager.concepts.keys()}

            # Encode prompt
            text_embeddings = pipe._encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=""
            )

            # Initialize latent
            latent = torch.randn(
                (1, pipe.unet.config.in_channels, 64, 64),
                device=device,
                dtype=text_embeddings.dtype
            )

            # Denoising loop
            pipe.scheduler.set_timesteps(args.num_inference_steps)
            timesteps = pipe.scheduler.timesteps

            for step_idx, timestep in enumerate(tqdm(timesteps, desc=f"Sample {sample_idx+1}")):
                # Expand latent for classifier-free guidance
                latent_model_input = torch.cat([latent] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

                # Predict noise
                with torch.no_grad():
                    noise_pred = pipe.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings
                    ).sample

                # CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)

                # Multi-concept guidance (BEFORE scheduler step)
                # Apply guidance to noise_pred instead of latent to preserve ODE/SDE structure
                if args.guidance_start_step <= step_idx < args.guidance_end_step:
                    # Detect harmful concepts
                    detections = manager.detect_harmful_concepts(
                        latent, timestep, current_step=step_idx
                    )

                    # Compute and apply combined guidance
                    if any(detections.values()):
                        combined_grad, concept_grads = manager.compute_combined_guidance(
                            latent=latent,
                            timestep=timestep,
                            current_step=step_idx,
                            detections=detections,
                            combination_strategy=args.combination_strategy
                        )

                        # Convert latent gradient to noise_pred space
                        # For DDIM/DDPM: noise_pred = (x_t - sqrt(alpha_t) * x_0) / sqrt(1 - alpha_t)
                        # Gradient in latent space needs to be converted to noise space
                        # Using: guidance_noise = -sigma_t * grad_latent

                        # Get sigma for this timestep
                        if hasattr(pipe.scheduler, 'sigmas') and len(pipe.scheduler.sigmas) > step_idx:
                            # For schedulers with explicit sigmas (e.g., Euler, DPM)
                            sigma_t = pipe.scheduler.sigmas[step_idx]
                        else:
                            # For DDIM/DDPM: sigma = sqrt((1 - alpha) / alpha)
                            alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep.item()]
                            sigma_t = ((1 - alpha_prod_t) / alpha_prod_t).sqrt()

                        # Convert latent gradient to noise_pred gradient
                        guidance_noise = -sigma_t * combined_grad

                        # Apply to noise_pred BEFORE scheduler step
                        noise_pred = noise_pred + guidance_noise

                # Compute previous sample
                latent = pipe.scheduler.step(noise_pred, timestep, latent).prev_sample

            # Decode to image
            with torch.no_grad():
                latent = 1 / 0.18215 * latent
                image = pipe.vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).astype(np.uint8)

            # Save image
            filename = f"prompt_{prompt_idx:04d}_sample_{sample_idx:02d}.png"
            save_image(image, output_dir / filename)

    # Print statistics
    print("\n" + "="*80)
    print("GENERATION STATISTICS")
    print("="*80)
    stats = manager.get_statistics()
    for concept_name, concept_stats in stats.items():
        print(f"\n{concept_name.upper()}:")
        print(f"  Total steps: {concept_stats['total_steps']}")
        print(f"  Detected steps: {concept_stats['detected_steps']}")
        print(f"  Detection rate: {concept_stats['detection_rate']:.1%}")

    # Print gradient conflict statistics
    if hasattr(manager, 'gradient_conflicts') and len(manager.gradient_conflicts) > 0:
        print("\n" + "-"*80)
        print("GRADIENT CONFLICT ANALYSIS")
        print("-"*80)
        conflicts = manager.gradient_conflicts

        # Count conflicts
        num_conflicts = sum(1 for c in conflicts if c['is_conflict'])
        total_multi_concept_steps = len(conflicts)

        print(f"\nTotal steps with both concepts detected: {total_multi_concept_steps}")
        print(f"Steps with gradient conflict (dot < 0): {num_conflicts}")
        print(f"Conflict rate: {num_conflicts/total_multi_concept_steps*100:.1f}%")

        # Average metrics
        avg_dot = sum(c['dot_product'] for c in conflicts) / len(conflicts)
        avg_cos = sum(c['cosine_similarity'] for c in conflicts) / len(conflicts)

        print(f"\nAverage dot product: {avg_dot:.2e}")
        print(f"Average cosine similarity: {avg_cos:.4f}")

        # Min/Max cosine similarity
        min_cos = min(c['cosine_similarity'] for c in conflicts)
        max_cos = max(c['cosine_similarity'] for c in conflicts)
        print(f"Cosine similarity range: [{min_cos:.4f}, {max_cos:.4f}]")

        # Save detailed conflict log
        conflict_log_path = output_dir / "gradient_conflicts.csv"
        import csv
        with open(conflict_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'dot_product', 'cosine_similarity',
                                                   'norm_nudity', 'norm_violence', 'is_conflict'])
            writer.writeheader()
            writer.writerows(conflicts)
        print(f"\nDetailed conflict log saved to: {conflict_log_path}")

    print("="*80)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"✓ Loaded {len(prompts)} prompts from {args.prompt_file}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # Load Stable Diffusion
    print(f"✓ Loading Stable Diffusion from {args.ckpt_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("✓ Stable Diffusion loaded")

    # Build multi-concept configuration
    concepts = {}

    # Nudity concept
    if args.nudity_enabled:
        print(f"✓ Loading nudity classifier from {args.nudity_classifier}...")
        # Nudity classifier: 3 classes (0=NotPeople, 1=Safe, 2=Full)
        nudity_classifier = load_discriminator(
            ckpt_path=args.nudity_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        # Keep classifier in float32 (same as single-concept code)

        # Verify all parameters are on device
        for name, param in nudity_classifier.named_parameters():
            if param.device.type == 'cpu':
                print(f"WARNING: Parameter {name} is still on CPU!")
                param.data = param.data.to(device)

        # Create threshold scheduler for nudity
        nudity_threshold_scheduler = None
        if args.use_adaptive_threshold:
            from geo_utils.selective_guidance_utils import ThresholdScheduler
            nudity_threshold_scheduler = ThresholdScheduler(
                strategy=args.threshold_strategy,
                start_value=args.nudity_threshold_start,
                end_value=args.nudity_threshold_end,
                total_steps=args.num_inference_steps
            )

        nudity_monitor = SelectiveGuidanceMonitor(
            classifier_model=nudity_classifier,
            harmful_class=args.nudity_harmful_class,
            safe_class=args.nudity_safe_class,
            harmful_threshold=args.nudity_harmful_threshold,
            use_adaptive_threshold=args.use_adaptive_threshold,
            threshold_scheduler=nudity_threshold_scheduler,
            device=device
        )

        nudity_guidance = SpatiallyMaskedGuidance(
            classifier_model=nudity_classifier,
            safe_class=args.nudity_safe_class,
            harmful_class=args.nudity_harmful_class,
            device=device,
            use_bidirectional=args.use_bidirectional,
            weight_scheduler=None,  # Will use nudity_weight_scheduler separately
            normalize_gradient=args.normalize_gradient,
            gradient_norm_type=args.gradient_norm_type
        )

        nudity_weight_scheduler = None
        if args.use_weight_scheduling:
            nudity_weight_scheduler = WeightScheduler(
                strategy=args.weight_strategy,
                start_step=args.guidance_start_step,
                end_step=args.guidance_end_step,
                start_weight=args.nudity_weight_start,
                end_weight=args.nudity_weight_end
            )

        concepts['nudity'] = {
            'monitor': nudity_monitor,
            'guidance': nudity_guidance,
            'weight_scheduler': nudity_weight_scheduler,
            'guidance_scale': args.nudity_guidance_scale,
            'harmful_scale': args.nudity_harmful_scale,
            'spatial_threshold': args.nudity_spatial_threshold,
            'enabled': True
        }
        print("✓ Nudity concept configured")

    # Violence concept
    if args.violence_enabled:
        print(f"✓ Loading violence classifier from {args.violence_classifier}...")
        # Violence classifier: 3 classes (0=NotRelevant, 1=Safe, 2=Full)
        violence_classifier = load_discriminator(
            ckpt_path=args.violence_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        # Keep classifier in float32 (same as single-concept code)

        # Verify all parameters are on device
        for name, param in violence_classifier.named_parameters():
            if param.device.type == 'cpu':
                print(f"WARNING: Parameter {name} is still on CPU!")
                param.data = param.data.to(device)

        # Create threshold scheduler for violence
        violence_threshold_scheduler = None
        if args.use_adaptive_threshold:
            from geo_utils.selective_guidance_utils import ThresholdScheduler
            violence_threshold_scheduler = ThresholdScheduler(
                strategy=args.threshold_strategy,
                start_value=args.violence_threshold_start,
                end_value=args.violence_threshold_end,
                total_steps=args.num_inference_steps
            )

        violence_monitor = SelectiveGuidanceMonitor(
            classifier_model=violence_classifier,
            harmful_class=args.violence_harmful_class,
            safe_class=args.violence_safe_class,
            harmful_threshold=args.violence_harmful_threshold,
            use_adaptive_threshold=args.use_adaptive_threshold,
            threshold_scheduler=violence_threshold_scheduler,
            device=device
        )

        violence_guidance = SpatiallyMaskedGuidance(
            classifier_model=violence_classifier,
            safe_class=args.violence_safe_class,
            harmful_class=args.violence_harmful_class,
            device=device,
            use_bidirectional=args.use_bidirectional,
            weight_scheduler=None,  # Will use violence_weight_scheduler separately
            normalize_gradient=args.normalize_gradient,
            gradient_norm_type=args.gradient_norm_type
        )

        violence_weight_scheduler = None
        if args.use_weight_scheduling:
            violence_weight_scheduler = WeightScheduler(
                strategy=args.weight_strategy,
                start_step=args.guidance_start_step,
                end_step=args.guidance_end_step,
                start_weight=args.violence_weight_start,
                end_weight=args.violence_weight_end
            )

        concepts['violence'] = {
            'monitor': violence_monitor,
            'guidance': violence_guidance,
            'weight_scheduler': violence_weight_scheduler,
            'guidance_scale': args.violence_guidance_scale,
            'harmful_scale': args.violence_harmful_scale,
            'spatial_threshold': args.violence_spatial_threshold,
            'enabled': True
        }
        print("✓ Violence concept configured")

    # VanGogh concept
    if args.vangogh_enabled:
        print(f"✓ Loading VanGogh classifier from {args.vangogh_classifier}...")
        # VanGogh classifier: 3 classes (0=NotRelevant, 1=Safe, 2=VanGogh style)
        vangogh_classifier = load_discriminator(
            ckpt_path=args.vangogh_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)

        # Verify all parameters are on device
        for name, param in vangogh_classifier.named_parameters():
            if param.device.type == 'cpu':
                print(f"WARNING: Parameter {name} is still on CPU!")
                param.data = param.data.to(device)

        # Create threshold scheduler for VanGogh
        vangogh_threshold_scheduler = None
        if args.use_adaptive_threshold:
            from geo_utils.selective_guidance_utils import ThresholdScheduler
            vangogh_threshold_scheduler = ThresholdScheduler(
                strategy=args.threshold_strategy,
                start_value=args.vangogh_threshold_start,
                end_value=args.vangogh_threshold_end,
                total_steps=args.num_inference_steps
            )

        vangogh_monitor = SelectiveGuidanceMonitor(
            classifier_model=vangogh_classifier,
            harmful_class=args.vangogh_harmful_class,
            safe_class=args.vangogh_safe_class,
            harmful_threshold=args.vangogh_harmful_threshold,
            use_adaptive_threshold=args.use_adaptive_threshold,
            threshold_scheduler=vangogh_threshold_scheduler,
            device=device
        )

        vangogh_guidance = SpatiallyMaskedGuidance(
            classifier_model=vangogh_classifier,
            safe_class=args.vangogh_safe_class,
            harmful_class=args.vangogh_harmful_class,
            device=device,
            use_bidirectional=args.use_bidirectional,
            weight_scheduler=None,  # Will use vangogh_weight_scheduler separately
            normalize_gradient=args.normalize_gradient,
            gradient_norm_type=args.gradient_norm_type
        )

        vangogh_weight_scheduler = None
        if args.use_weight_scheduling:
            vangogh_weight_scheduler = WeightScheduler(
                strategy=args.weight_strategy,
                start_step=args.guidance_start_step,
                end_step=args.guidance_end_step,
                start_weight=args.vangogh_weight_start,
                end_weight=args.vangogh_weight_end
            )

        concepts['vangogh'] = {
            'monitor': vangogh_monitor,
            'guidance': vangogh_guidance,
            'weight_scheduler': vangogh_weight_scheduler,
            'guidance_scale': args.vangogh_guidance_scale,
            'harmful_scale': args.vangogh_harmful_scale,
            'spatial_threshold': args.vangogh_spatial_threshold,
            'enabled': True
        }
        print("✓ VanGogh concept configured")

    # Create multi-concept manager
    manager = MultiConceptGuidanceManager(concepts=concepts, device=device)

    # Generate images
    generate_with_multi_concept_guidance(
        pipe=pipe,
        prompts=prompts,
        manager=manager,
        args=args,
        output_dir=args.output_dir
    )

    print("\n✅ Multi-concept generation complete!")
    print(f"✅ Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
