#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Unlearning with CLASSIFIER-GUIDED OUTPUT MASKING

Novel Approach:
  1. Use Grad-CAM to identify nude regions in latent space at each diffusion step
  2. Create spatial masks at multiple resolutions (64×64, 32×32, 16×16, 8×8)
  3. Suppress cross-attention OUTPUT at nude spatial positions
  4. Prevents text semantics from being injected into nude regions

How Cross-Attention Works:
  - Query: from latent spatial positions [B, 4096, dim] (64×64 flattened)
  - Key/Value: from CLIP text embeddings [B, 77, dim]
  - Attention: Query × Key^T → attention weights [B, 4096, 77]
  - Output: attention weights × Value → [B, 4096, dim]
  - Output is reshaped and injected back into latent

Why OUTPUT Masking?
  - We want to suppress what gets injected WHERE in the latent
  - Output masking directly controls spatial injection of text semantics
  - Mask shape: [B, spatial_size, spatial_size] applied to [B, spatial_size, spatial_size, C]

Masking Strategies:
  - HARD: Binary mask (nude=0, safe=1) → Complete suppression
  - SOFT: Weighted mask (nude=1-strength, safe=1) → Gradual suppression
  - ADVERSARIAL: Can go negative (experimental) → Active reversal

Technical Flow:
  1. Callback receives current latent at each diffusion step
  2. Attention processor computes Grad-CAM heatmap for nude class (class 2)
  3. Heatmap is thresholded to create binary/soft mask
  4. Mask is downsampled to match different U-Net resolutions
  5. During cross-attention forward pass:
     - Compute attention output normally
     - Reshape output to spatial format [B, H, W, C]
     - Apply spatial mask: output_spatial = output_spatial * mask
     - Reshape back and continue
  6. Result: Text semantics are blocked from nude regions

Combination with Other Techniques:
  - Can be combined with attention score suppression (harm_suppress)
  - Can be combined with classifier guidance (classifier_guidance)
  - All three techniques work at different stages of the pipeline

Innovation:
  - Direct spatial control over cross-attention output injection
  - Classifier-guided dynamic masking per diffusion step
  - No need for model fine-tuning or retraining
"""

import os
import random
from argparse import ArgumentParser
from functools import partial
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.attention_processor import AttnProcessor2_0, Attention

from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import GuidanceModel
from geo_utils.classifier_interpretability import ClassifierGradCAM

import numpy as np
from typing import List, Optional, Tuple


# =========================
# General Classifier Guidance (Direct gradient-based)
# =========================
class GeneralClassifierGuidance:
    """
    General classifier guidance using direct gradient computation.

    Simpler and more flexible than GuidanceModel wrapper.
    Supports both unidirectional and bidirectional guidance.
    """

    def __init__(
        self,
        classifier_model,
        safe_class: int = 1,
        harmful_class: int = 2,
        use_bidirectional: bool = False,
        device: str = "cuda"
    ):
        """
        Args:
            classifier_model: Loaded classifier model
            safe_class: Target class for guidance (1 = clothed)
            harmful_class: Class to avoid (2 = nude)
            use_bidirectional: If True, pull toward safe + push from harmful
            device: Device to run on
        """
        self.classifier = classifier_model
        self.safe_class = safe_class
        self.harmful_class = harmful_class
        self.use_bidirectional = use_bidirectional
        self.device = device

        # Get classifier dtype from its parameters
        self.classifier_dtype = next(self.classifier.parameters()).dtype

    def compute_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute classifier gradient.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Timestep value
            guidance_scale: Gradient scale for safe direction
            harmful_scale: Relative scale for harmful repulsion

        Returns:
            gradient: [B, 4, H, W] gradient to add to latent
        """
        with torch.enable_grad():
            # Ensure latent requires grad and matches classifier dtype
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            # Ensure timestep is tensor
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B)

            if self.use_bidirectional:
                # Bidirectional guidance
                # 1. Gradient toward SAFE class (pull)
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, timestep)
                safe_logit = logits_safe[:, self.safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                # 2. Gradient toward HARMFUL class (to push opposite direction)
                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, timestep)
                harmful_logit = logits_harmful[:, self.harmful_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                # Combine: pull toward safe, push away from harmful
                grad = grad_safe - harmful_scale * grad_harmful
            else:
                # Unidirectional guidance (toward safe class only)
                logits = self.classifier(latent_input, timestep)
                safe_logit = logits[:, self.safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        # Scale
        grad = grad * guidance_scale

        # Convert back to original latent dtype
        grad = grad.to(dtype=latent.dtype)

        return grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Apply general classifier guidance to latent.

        Args:
            latent: [B, 4, H, W] current latent
            timestep: Current timestep
            guidance_scale: Gradient scale
            harmful_scale: Harmful repulsion scale (if bidirectional)

        Returns:
            guided_latent: [B, 4, H, W] latent after guidance
        """
        grad = self.compute_gradient(latent, timestep, guidance_scale, harmful_scale)
        guided_latent = latent + grad
        return guided_latent


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Classifier-Guided Value Masking for Unlearning")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/classifier_masked", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Harmful concept suppression
    parser.add_argument("--harm_suppress", action="store_true")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt")
    parser.add_argument("--base_tau", type=float, default=0.15)
    parser.add_argument("--harm_gamma_start", type=float, default=40.0)
    parser.add_argument("--harm_gamma_end", type=float, default=0.5)

    # Classifier Guidance (Old wrapper-based approach)
    parser.add_argument("--classifier_guidance", action="store_true",
                        help="Enable classifier guidance (old GuidanceModel approach)")
    parser.add_argument("--classifier_config", type=str,
                        default="./configs/models/time_dependent_discriminator.yaml")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--guidance_start_step", type=int, default=1)
    parser.add_argument("--target_class", type=int, default=1)

    # General Classifier Guidance (NEW: Direct gradient-based approach)
    parser.add_argument("--general_cg", action="store_true",
                        help="Enable general classifier guidance (direct gradient approach)")
    parser.add_argument("--general_cg_scale", type=float, default=5.0,
                        help="Gradient scale for general CG")
    parser.add_argument("--general_cg_safe_class", type=int, default=1,
                        help="Safe class for guidance target (1 = clothed)")
    parser.add_argument("--general_cg_harmful_class", type=int, default=2,
                        help="Harmful class to avoid (2 = nude)")
    parser.add_argument("--general_cg_use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance (pull to safe + push from harmful)")
    parser.add_argument("--general_cg_harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale (relative to general_cg_scale)")
    parser.add_argument("--general_cg_start_step", type=int, default=0,
                        help="Step to start general CG")
    parser.add_argument("--general_cg_end_step", type=int, default=50,
                        help="Step to end general CG")

    # NEW: Value Masking Parameters
    parser.add_argument("--value_masking", action="store_true",
                        help="Enable classifier-guided value masking")
    parser.add_argument("--mask_strategy", type=str, default="soft",
                        choices=["hard", "soft", "adversarial"],
                        help="Masking strategy: hard (zero), soft (weighted), adversarial (reverse)")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Grad-CAM threshold for masking (0-1)")
    parser.add_argument("--mask_percentile", type=float, default=0.8,
                        help="Top percentile to mask (alternative to threshold)")
    parser.add_argument("--use_percentile", action="store_true",
                        help="Use percentile instead of absolute threshold")
    parser.add_argument("--mask_strength", type=float, default=1.0,
                        help="Masking strength (0-1, only for soft/adversarial)")
    parser.add_argument("--mask_start_step", type=int, default=5,
                        help="Step to start value masking")
    parser.add_argument("--mask_end_step", type=int, default=45,
                        help="Step to end value masking")

    # DEBUG
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_prompts", action="store_true",
                        help="Show per-token analysis")
    parser.add_argument("--debug_steps", action="store_true",
                        help="Show per-step masking statistics")
    parser.add_argument("--save_heatmaps", action="store_true",
                        help="Save Grad-CAM heatmaps for debugging")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def save_image(image, filename, root="output_img"):
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def schedule_linear(step: int, num_steps: int, start_val: float, end_val: float) -> float:
    t = step / max(1, num_steps - 1)
    return start_val * (1.0 - t) + end_val * t


# =========================
# Vector Building
# =========================
@torch.no_grad()
def build_harm_vector(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    if not concepts:
        return None

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    tokens = tokenizer(
        concepts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-2]
    hidden_states = F.normalize(hidden_states, dim=-1)

    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask.bool()

    content_mask = attention_mask.clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    denom = content_mask.sum(dim=1, keepdim=True).clamp(min=1)
    vectors = (hidden_states * content_mask.unsqueeze(-1)).sum(dim=1) / denom
    harm_vector = F.normalize(vectors.mean(dim=0), dim=-1)

    return harm_vector


# =========================
# Classifier-Guided Value Masking Processor
# =========================
class ClassifierMaskedAttnProcessor(AttnProcessor2_0):
    """
    Attention processor with classifier-guided value masking.

    Key idea:
    1. Get Grad-CAM heatmap from classifier on current latent
    2. Identify nude regions (high activation)
    3. Mask corresponding positions in value vectors
    4. Prevents nude features from being attended to
    """

    def __init__(self,
                 classifier_model=None,
                 harm_vector: Optional[torch.Tensor] = None,
                 tau: float = 0.15,
                 gamma: float = 1.0,
                 mask_strategy: str = "soft",
                 mask_threshold: float = 0.5,
                 mask_strength: float = 1.0,
                 use_percentile: bool = False,
                 mask_percentile: float = 0.8,
                 debug: bool = False):
        super().__init__()

        self.classifier = classifier_model
        self.tau = tau
        self.gamma = gamma
        self.mask_strategy = mask_strategy
        self.mask_threshold = mask_threshold
        self.mask_strength = mask_strength
        self.use_percentile = use_percentile
        self.mask_percentile = mask_percentile
        self.debug = debug

        # Store harm vector
        if harm_vector is None or harm_vector.numel() == 0:
            self._harm_vector = None
        else:
            self._harm_vector = F.normalize(harm_vector.detach().float(), dim=-1).cpu()

        # Grad-CAM instance (created lazily)
        self._gradcam = None

        # Current state
        self.current_step = 0
        self.current_timestep = None
        self.current_latent = None  # Store current latent from callback
        self.enable_masking = False
        self._spatial_masks = {}  # Cache masks by resolution

        # Debug stats
        self.debug_stats = []

    @property
    def harm_vector(self):
        return self._harm_vector

    def set_gamma(self, gamma: float):
        self.gamma = float(gamma)

    def set_step(self, step: int, timestep: int, latent: Optional[torch.Tensor] = None):
        """Update current step and cache spatial masks if latent changed."""
        self.current_step = step
        self.current_timestep = timestep

        # Update latent and recompute masks if needed
        if latent is not None and self.enable_masking:
            # Check if latent actually changed
            needs_update = (
                self.current_latent is None or
                not torch.equal(latent, self.current_latent)
            )

            if needs_update:
                # Clone but keep gradients for Grad-CAM computation
                # The ClassifierGradCAM will handle requires_grad internally
                self.current_latent = latent.clone()
                self._spatial_masks = {}  # Clear cache

                # Precompute masks for all resolutions
                self._precompute_spatial_masks()

    def set_masking_enabled(self, enabled: bool):
        self.enable_masking = enabled
        if not enabled:
            self._spatial_masks = {}  # Clear cache when disabled
            self.current_latent = None

    def _precompute_spatial_masks(self):
        """
        Precompute masks for different spatial resolutions.
        UNet has multiple resolutions: 64x64, 32x32, 16x16, 8x8
        """
        if self.current_latent is None or self.classifier is None:
            return

        device = self.current_latent.device
        B = self.current_latent.shape[0]

        # Initialize Grad-CAM if needed
        if self._gradcam is None:
            self._gradcam = ClassifierGradCAM(
                self.classifier,
                target_layer_name="encoder_model.middle_block.2"
            )

        try:
            # Compute Grad-CAM heatmap at 64x64 resolution
            # Need to enable gradients for Grad-CAM computation
            with torch.enable_grad():
                timestep_tensor = torch.tensor([self.current_timestep], device=device, dtype=torch.long)
                if timestep_tensor.shape[0] != B:
                    timestep_tensor = timestep_tensor.repeat(B)

                heatmap, info = self._gradcam.generate_heatmap(
                    self.current_latent,
                    timestep_tensor,
                    target_class=2,  # Nude class
                    normalize=True
                )
            # heatmap: [B, 64, 64], values in [0, 1]

            # Determine threshold
            if self.use_percentile:
                # Use top percentile
                thresholds = []
                for b in range(B):
                    h = heatmap[b].flatten()
                    k = int(h.numel() * self.mask_percentile)
                    if k > 0:
                        threshold = torch.topk(h, k=k)[0][-1].item()
                    else:
                        threshold = h.max().item()
                    thresholds.append(threshold)
                threshold_tensor = torch.tensor(thresholds, device=device).view(B, 1, 1)
                nude_mask = (heatmap >= threshold_tensor).float()
            else:
                # Use absolute threshold
                nude_mask = (heatmap >= self.mask_threshold).float()

            # Apply masking strategy to create keep_mask
            if self.mask_strategy == "hard":
                # Hard mask: completely zero out nude regions
                keep_mask_64 = 1.0 - nude_mask
            elif self.mask_strategy == "soft":
                # Soft mask: reduce contribution by strength
                keep_mask_64 = 1.0 - (nude_mask * self.mask_strength)
            elif self.mask_strategy == "adversarial":
                # Adversarial: reverse gradient direction
                # Higher multiplier = stronger reversal (2.0 = full reversal, 3.0+ = over-reversal)
                keep_mask_64 = 1.0 - (nude_mask * self.mask_strength * 3.0)
            else:
                keep_mask_64 = torch.ones_like(nude_mask)

            # Store 64x64 mask
            self._spatial_masks[64] = keep_mask_64  # [B, 64, 64]

            # Downsample to other resolutions
            for size in [32, 16, 8]:
                mask_resized = F.interpolate(
                    keep_mask_64.unsqueeze(1),  # [B, 1, 64, 64]
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [B, size, size]
                self._spatial_masks[size] = mask_resized

            # Debug stats
            if self.debug:
                num_masked = nude_mask.sum().item()
                total_pixels = nude_mask.numel()
                avg_heatmap = heatmap.mean().item()
                max_heatmap = heatmap.max().item()

                self.debug_stats.append({
                    'step': self.current_step,
                    'num_masked': num_masked,
                    'total_pixels': total_pixels,
                    'mask_ratio': num_masked / max(1, total_pixels),
                    'avg_heatmap': avg_heatmap,
                    'max_heatmap': max_heatmap,
                    'strategy': self.mask_strategy,
                    'threshold': threshold_tensor.mean().item() if self.use_percentile else self.mask_threshold,
                })

        except Exception as e:
            if self.debug:
                print(f"[WARNING] Grad-CAM computation failed: {e}")
            self._spatial_masks = {}

    def get_spatial_mask(self, spatial_size: int) -> Optional[torch.Tensor]:
        """
        Get precomputed spatial mask for given resolution.

        Args:
            spatial_size: Spatial resolution (64, 32, 16, or 8)

        Returns:
            mask: [B, spatial_size, spatial_size] or None
        """
        if not self.enable_masking or spatial_size not in self._spatial_masks:
            return None
        return self._spatial_masks[spatial_size]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = hidden_states.shape
        is_cross_attn = encoder_hidden_states is not None

        # Pre-processing
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q, K, V projections
        query = attn.to_q(hidden_states)
        if is_cross_attn:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)  # This is where CLIP features become values!
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # ==========================================
        # Reshape for multi-head attention (BEFORE masking)
        # ==========================================
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

        # Apply harmful concept suppression (original method)
        if is_cross_attn and self._harm_vector is not None:
            device = scores.device
            harm_vec = self._harm_vector.to(device)

            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            num_heads = scores.shape[0] // B

            context_normalized = F.normalize(encoder_hidden_states, dim=-1)
            harm_normalized = F.normalize(harm_vec, dim=-1)
            cosine_sim = torch.einsum("bkd,d->bk", context_normalized, harm_normalized)

            suppress_mask = (cosine_sim >= self.tau)

            if suppress_mask.any():
                weight = cosine_sim.clamp(min=0.0) * suppress_mask.float()
                weight_expanded = weight[:, None, :].expand(B, Q, K).repeat_interleave(num_heads, dim=0)
                scores = scores - (weight_expanded * self.gamma)

        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        # Softmax
        attn_probs = F.softmax(scores, dim=-1)

        # Dropout
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        # Output
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # ==========================================
        # NEW: Classifier-Guided Output Masking ⭐
        # ==========================================
        if is_cross_attn and self.enable_masking:
            # hidden_states after attention: [B, H*W, C]
            B = hidden_states.shape[0]
            HW = hidden_states.shape[1]
            C = hidden_states.shape[2]
            spatial_size = int(np.sqrt(HW))

            if spatial_size * spatial_size == HW:
                # Get precomputed nude mask
                spatial_mask = self.get_spatial_mask(spatial_size)

                if spatial_mask is not None:
                    # Reshape output to spatial format
                    # [B, HW, C] -> [B, spatial_size, spatial_size, C]
                    output_spatial = hidden_states.view(B, spatial_size, spatial_size, C)

                    # Apply mask: suppress nude regions
                    # spatial_mask: [B, spatial_size, spatial_size]
                    mask_expanded = spatial_mask.unsqueeze(-1)  # [B, spatial_size, spatial_size, 1]

                    # Apply masking strategy
                    if self.mask_strategy == "hard":
                        # Hard mask: completely zero out nude regions
                        output_spatial = output_spatial * mask_expanded
                    elif self.mask_strategy == "soft":
                        # Soft mask: weighted suppression
                        output_spatial = output_spatial * mask_expanded
                    elif self.mask_strategy == "adversarial":
                        # Adversarial: can go negative
                        output_spatial = output_spatial * mask_expanded

                    # Reshape back to [B, HW, C]
                    hidden_states = output_spatial.view(B, HW, C)

        # Final projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def print_debug_stats(self):
        if self.debug and self.debug_stats:
            recent = self.debug_stats[-1]
            print(f"  [MASK Step {recent['step']:02d}] "
                  f"Masked: {recent['num_masked']}/{recent['total_pixels']} ({100*recent['mask_ratio']:.1f}%) | "
                  f"Heatmap: avg={recent['avg_heatmap']:.3f}, max={recent['max_heatmap']:.3f}")


# =========================
# Main Generation Loop
# =========================
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*100}")
    print(f"[INFO] Classifier-Guided Value Masking for Machine Unlearning")
    print(f"{'='*100}")
    print(f"[INFO] Loading model from {args.ckpt_path}")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None
    ).to(device)
    print(f"[INFO] Model loaded on device: {pipe.device}")

    # Load prompts
    with open(os.path.expanduser(args.prompt_file), "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(prompts)} prompts")

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load classifier for value masking, old CG, or general CG
    classifier_model = None
    if args.value_masking or args.classifier_guidance or args.general_cg:
        from geo_models.classifier.classifier import load_discriminator
        print(f"\n[INFO] Loading classifier from {args.classifier_ckpt}")
        classifier_model = load_discriminator(
            ckpt_path=args.classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        classifier_model.eval()
        print(f"[INFO] Classifier loaded")

    # Setup harmful concept vector
    harm_vector = None
    harm_concepts = []
    if args.harm_suppress:
        harm_file = os.path.expanduser(args.harm_concepts_file)
        if os.path.isfile(harm_file):
            with open(harm_file, "r") as f:
                harm_concepts = [line.strip() for line in f if line.strip()]
            print(f"\n[INFO] Building harmful concept vector from {len(harm_concepts)} concepts")
            harm_vector = build_harm_vector(pipe, harm_concepts)

    # Setup attention processor with value masking
    attn_processor = ClassifierMaskedAttnProcessor(
        classifier_model=classifier_model if args.value_masking else None,
        harm_vector=harm_vector,
        tau=args.base_tau,
        gamma=args.harm_gamma_start,
        mask_strategy=args.mask_strategy,
        mask_threshold=args.mask_threshold,
        mask_strength=args.mask_strength,
        use_percentile=args.use_percentile,
        mask_percentile=args.mask_percentile,
        debug=args.debug
    )
    pipe.unet.set_attn_processor(attn_processor)

    print(f"\n[INFO] Value Masking Configuration:")
    print(f"  - Enabled: {args.value_masking}")
    print(f"  - Strategy: {args.mask_strategy}")
    print(f"  - Threshold: {args.mask_threshold if not args.use_percentile else f'top {args.mask_percentile*100:.0f}%'}")
    print(f"  - Strength: {args.mask_strength}")
    print(f"  - Active steps: {args.mask_start_step} → {args.mask_end_step}")

    # Setup OLD classifier guidance (GuidanceModel wrapper)
    guidance_model = None
    if args.classifier_guidance:
        guidance_model = GuidanceModel(
            pipe,
            args.classifier_config,
            args.classifier_ckpt,
            1,
            device
        )
        print(f"\n[INFO] OLD Classifier Guidance enabled (scale={args.guidance_scale})")

    # Setup GENERAL classifier guidance (Direct gradient-based)
    general_cg = None
    if args.general_cg:
        general_cg = GeneralClassifierGuidance(
            classifier_model=classifier_model,
            safe_class=args.general_cg_safe_class,
            harmful_class=args.general_cg_harmful_class,
            use_bidirectional=args.general_cg_use_bidirectional,
            device=device
        )
        guidance_mode = "bidirectional" if args.general_cg_use_bidirectional else "unidirectional"
        print(f"\n[INFO] General Classifier Guidance enabled ({guidance_mode})")
        print(f"  Scale: {args.general_cg_scale}")
        print(f"  Safe class: {args.general_cg_safe_class}")
        print(f"  Harmful class: {args.general_cg_harmful_class}")
        if args.general_cg_use_bidirectional:
            print(f"  Harmful repulsion scale: {args.general_cg_harmful_scale}")
        print(f"  Active steps: {args.general_cg_start_step} → {args.general_cg_end_step}")

    # Generation callback
    def callback_on_step_end(
        diffusion_pipeline,
        step,
        timestep,
        callback_kwargs,
        **kwargs
    ):
        # Get current latent from callback_kwargs
        latents = callback_kwargs.get("latents")

        # Update processor state with current latent for Grad-CAM computation
        attn_processor.set_step(step, int(timestep), latent=latents)

        # Enable/disable masking based on step range
        enable_mask = (args.mask_start_step <= step <= args.mask_end_step)
        attn_processor.set_masking_enabled(enable_mask)

        # Update gamma schedule
        gamma = schedule_linear(step, args.num_inference_steps, args.harm_gamma_start, args.harm_gamma_end)
        attn_processor.set_gamma(gamma)

        # Apply OLD classifier guidance (GuidanceModel)
        if guidance_model is not None and step >= args.guidance_start_step:
            callback_kwargs = guidance_model.guidance(
                diffusion_pipeline,
                callback_kwargs,
                step,
                timestep,
                args.guidance_scale,
                target_class=args.target_class
            )
            latents = callback_kwargs.get("latents")  # Update latents after old CG

        # Apply GENERAL classifier guidance (Direct gradient-based)
        if general_cg is not None and (args.general_cg_start_step <= step <= args.general_cg_end_step):
            # Apply general CG to current latents
            guided_latents = general_cg.apply_guidance(
                latent=latents,
                timestep=timestep,
                guidance_scale=args.general_cg_scale,
                harmful_scale=args.general_cg_harmful_scale
            )
            callback_kwargs["latents"] = guided_latents

        if args.debug:
            attn_processor.print_debug_stats()

        return callback_kwargs

    # Generate
    print(f"\n{'='*100}")
    print(f"[INFO] Starting generation...")
    print(f"{'='*100}\n")

    for idx, prompt in enumerate(prompts):
        print(f"\n[PROMPT {idx + 1}/{len(prompts)}] {prompt}")

        attn_processor.debug_stats = []

        with torch.enable_grad():
            output = pipe(
                prompt=prompt,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=512,
                width=512,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents", "noise_pred", "prev_latents"],
                num_images_per_prompt=args.nsamples,
            )

        for sample_idx, image in enumerate(output.images):
            filename = f"prompt_{idx+1:04d}_sample_{sample_idx+1}.png"
            save_image(image, filename, root=output_dir)

        print(f"[INFO] Saved {len(output.images)} image(s)")

    print(f"\n{'='*100}")
    print(f"[INFO] Generation complete! Images saved to {output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
