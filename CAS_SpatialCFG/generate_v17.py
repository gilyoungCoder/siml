#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IP-Adapter Image Projection + CAS Safe Generation v17

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau (same as v6)
  WHERE (Probe + optional noise CAS fusion): spatial mask built from:
        --probe_source {text, image, both} controls WHICH probe(s) to run
        --where_mode   {probe_only, fused, noise_only} controls HOW to combine
                       the probe mask with the noise-CAS mask
  HOW: Same guidance modes as v6 (hybrid recommended)

probe_source choices:
  text    -- v6-style text keyword cross-attention probe
  image   -- IP-Adapter projected CLIP image cross-attention probe
  both    -- union (max) of text + image probe masks  [default]

where_mode choices:
  probe_only  -- use probe_source mask only (no noise CAS)
  fused       -- max(probe_source_mask, noise_CAS_mask)
  noise_only  -- noise-based spatial CAS mask only (probe_source ignored)

Key innovations vs v13:
  - WHERE probe uses IP-Adapter's pre-trained Resampler (Perceiver) to project
    CLIP ViT-L/14 image features into the UNet's cross-attention K/V space
  - IP-Adapter was trained on millions of image-text pairs, so the projection
    produces high-quality cross-attention-compatible representations
  - SD1.4 and SD1.5 share identical UNet architecture, so IP-Adapter SD1.5
    weights are fully compatible
  - Still "training-free" for our safety task (we use pre-trained IP-Adapter as-is)

Fallback:
  If IP-Adapter loading fails, falls back to a 2-layer MLP projection with
  Xavier initialization (v17_simple mode). This is documented and provides a
  non-degenerate mapping better than raw CLS-repeat.

Usage:
    # 1. Both probes fused with noise CAS (new default):
    CUDA_VISIBLE_DEVICES=0 python generate_v17.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v17/ringabell_both_fused \
        --clip_embeddings exemplars/sd14/clip_exemplar_embeddings.pt \
        --probe_source both --where_mode fused \
        --guide_mode hybrid --safety_scale 1.5

    # 2. Image probe only, no noise CAS:
    CUDA_VISIBLE_DEVICES=0 python generate_v17.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v17/ringabell_ip_probe_only \
        --clip_embeddings exemplars/sd14/clip_exemplar_embeddings.pt \
        --probe_source image --where_mode probe_only \
        --guide_mode hybrid --safety_scale 1.5

    # 3. Text probe only (v6 style):
    CUDA_VISIBLE_DEVICES=0 python generate_v17.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v17/ringabell_text_only \
        --probe_source text --where_mode probe_only \
        --guide_mode hybrid --safety_scale 1.5
"""

import os
import sys
import json
import math
import random
import csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from diffusers import StableDiffusionPipeline, DDIMScheduler

from attention_probe import (
    AttentionProbeStore,
    precompute_target_keys,
    register_attention_probe,
    register_dual_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)
from concept_pack_loader import load_concept_pack, load_multiple_packs, get_combined_target_words


# =============================================================================
# IP-Adapter Image Projection (Resampler) Loader
# =============================================================================
def load_ip_adapter_image_proj(
    ip_adapter_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Optional[nn.Module]:
    """
    Load IP-Adapter's image_proj_model (Resampler/Perceiver) from weights.

    The image_proj_model maps CLIP image features [B, 257, 1024] (ViT-H/14 for
    IP-Adapter SD1.5, or [B, 1, 768] for plus-variant) into cross-attention-
    compatible tokens [B, num_tokens, 768].

    We use diffusers' built-in IP-Adapter support to load the projection module.

    Args:
        ip_adapter_path: Path to ip-adapter .bin file. If None, attempts to
                         use h94/IP-Adapter from HuggingFace Hub.
        device: Target device
        dtype: Target dtype

    Returns:
        image_proj_model: The loaded projection module, or None if loading fails.
    """
    try:
        from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
        # Load a temporary pipeline to extract the image_proj_model
        print("  [v17] Loading IP-Adapter image projection module...")

        # We load a lightweight pipeline just to get the IP-Adapter projection
        temp_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=dtype,
            safety_checker=None,
        )

        if ip_adapter_path and os.path.isfile(ip_adapter_path):
            # Load from local file
            print(f"  [v17] Loading from local path: {ip_adapter_path}")
            # Extract image_proj state dict from the IP-Adapter checkpoint
            state_dict = torch.load(ip_adapter_path, map_location="cpu")
            if "image_proj" in state_dict:
                image_proj_sd = state_dict["image_proj"]
            else:
                image_proj_sd = {
                    k.replace("image_proj_model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("image_proj")
                }

            # Try diffusers load_ip_adapter
            try:
                temp_pipe.load_ip_adapter(
                    ip_adapter_path,
                    subfolder="",
                    weight_name=os.path.basename(ip_adapter_path),
                )
            except Exception:
                # If local file doesn't work with load_ip_adapter, try hub
                temp_pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                )
        else:
            # Load from HuggingFace Hub
            print("  [v17] Loading IP-Adapter from h94/IP-Adapter hub...")
            temp_pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )

        # Extract the image projection model
        image_proj = temp_pipe.unet.encoder_hid_proj
        if image_proj is None:
            print("  [v17] WARNING: encoder_hid_proj is None after loading IP-Adapter")
            del temp_pipe
            return None

        # Move to target device/dtype and detach from the temp pipeline
        image_proj = image_proj.to(device=device, dtype=dtype)
        image_proj.eval()
        for param in image_proj.parameters():
            param.requires_grad_(False)

        print(f"  [v17] IP-Adapter image_proj loaded: {type(image_proj).__name__}")
        num_params = sum(p.numel() for p in image_proj.parameters())
        print(f"  [v17] Parameters: {num_params / 1e6:.1f}M")

        # Cleanup temp pipeline to free memory
        del temp_pipe
        torch.cuda.empty_cache()

        return image_proj

    except Exception as e:
        print(f"  [v17] WARNING: Failed to load IP-Adapter: {e}")
        print("  [v17] Will fall back to simple MLP projection (v17_simple)")
        return None


# =============================================================================
# Fallback: Simple MLP Projection (v17_simple)
# =============================================================================
class SimpleMLP(nn.Module):
    """
    Fallback 2-layer MLP projection for when IP-Adapter loading fails.

    Maps CLIP image features [B, clip_dim] to cross-attention compatible
    tokens [B, num_tokens, cross_attn_dim] with Xavier initialization.

    This is NOT a trained projection but provides a non-degenerate mapping
    that is better than raw CLS-repeat.
    """

    def __init__(
        self,
        clip_dim: int = 768,
        cross_attn_dim: int = 768,
        num_tokens: int = 4,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attn_dim * num_tokens),
        )
        # Xavier init for non-degenerate initial mapping
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, clip_dim] CLS token features

        Returns:
            [B, num_tokens, cross_attn_dim]
        """
        B = x.shape[0]
        out = self.proj(x)  # [B, cross_attn_dim * num_tokens]
        return out.view(B, self.num_tokens, -1)


# =============================================================================
# Project CLIP embeddings through IP-Adapter or fallback
# =============================================================================
def project_clip_to_crossattn(
    clip_image_features: torch.Tensor,
    image_proj_model: Optional[nn.Module],
    fallback_mlp: Optional[SimpleMLP],
    num_tokens: int = 4,
    cross_attn_dim: int = 768,
    tokenizer_max_length: int = 77,
) -> torch.Tensor:
    """
    Project CLIP image features into cross-attention-compatible embeddings.

    Handles three cases:
    1. IP-Adapter Resampler: Uses the pre-trained projection module
    2. SimpleMLP fallback: Uses the Xavier-initialized MLP
    3. Neither available: CLS-repeat fallback (same as v13)

    The output is padded to [1, 77, cross_attn_dim] to match text embedding shape,
    with projected tokens at positions 1..num_tokens and zeros elsewhere.

    Args:
        clip_image_features: CLIP image features.
            For IP-Adapter: [B, 257, 1024] (ViT-H patch+CLS) or [B, 1, 768] (CLS only)
            For fallback: [B, 768] (CLS token)
        image_proj_model: IP-Adapter Resampler module or None
        fallback_mlp: SimpleMLP fallback or None
        num_tokens: Number of output tokens
        cross_attn_dim: UNet cross-attention dimension (768 for SD1.4)
        tokenizer_max_length: Text token sequence length (77 for CLIP)

    Returns:
        projected_embeds: [1, 77, cross_attn_dim] with projected tokens placed
    """
    device = clip_image_features.device
    dtype = clip_image_features.dtype

    if image_proj_model is not None:
        # IP-Adapter Resampler projection
        with torch.no_grad():
            # The IP-Adapter image_proj expects [B, seq, dim] input
            if clip_image_features.dim() == 2:
                # [B, dim] -> [B, 1, dim]
                feat_in = clip_image_features.unsqueeze(1)
            else:
                feat_in = clip_image_features

            # IP-Adapter's image projection (Resampler/ImageProjModel)
            projected = image_proj_model(feat_in)  # [B, num_tokens, cross_attn_dim]

            if projected.shape[-1] != cross_attn_dim:
                # If dimensions don't match (e.g., IP-Adapter uses different dim),
                # apply a linear projection
                projected = F.linear(
                    projected,
                    torch.eye(cross_attn_dim, projected.shape[-1],
                              device=device, dtype=dtype),
                )

    elif fallback_mlp is not None:
        # SimpleMLP fallback
        with torch.no_grad():
            if clip_image_features.dim() == 3:
                # Take CLS token if full patch sequence provided
                feat_in = clip_image_features[:, 0]  # [B, dim]
            else:
                feat_in = clip_image_features
            projected = fallback_mlp(feat_in.to(fallback_mlp.proj[0].weight.dtype))

    else:
        # Last resort: CLS-repeat (same as v13)
        if clip_image_features.dim() == 3:
            cls_feat = clip_image_features[:, 0]  # [B, dim]
        else:
            cls_feat = clip_image_features

        if cls_feat.shape[-1] != cross_attn_dim:
            cls_feat = cls_feat[..., :cross_attn_dim]

        projected = cls_feat.unsqueeze(1).expand(-1, num_tokens, -1)

    # Ensure correct shape
    actual_tokens = projected.shape[1]
    if actual_tokens > num_tokens:
        projected = projected[:, :num_tokens]
    elif actual_tokens < num_tokens:
        pad = torch.zeros(
            projected.shape[0], num_tokens - actual_tokens, projected.shape[-1],
            device=device, dtype=dtype,
        )
        projected = torch.cat([projected, pad], dim=1)

    # Pad to [1, 77, cross_attn_dim] format (matching text embedding shape)
    B = projected.shape[0]
    output = torch.zeros(B, tokenizer_max_length, cross_attn_dim,
                         device=device, dtype=dtype)
    # Place projected tokens at positions 1..num_tokens (0 is BOS-equivalent)
    output[:, 1:1 + num_tokens] = projected.to(dtype)

    return output


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN) -- same as v6
# =============================================================================
class GlobalCAS:
    """Detect harmful prompt alignment via cosine(d_prompt, d_target)."""

    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target):
        d_prompt = (eps_prompt - eps_null).reshape(1, -1).float()
        d_target = (eps_target - eps_null).reshape(1, -1).float()
        cas = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if math.isnan(cas) or math.isinf(cas):
            return 0.0, self.triggered if self.sticky else False

        if self.sticky and self.triggered:
            return cas, True

        if cas > self.threshold:
            if self.sticky:
                self.triggered = True
            return cas, True

        return cas, False


# =============================================================================
# Spatial CAS (noise-based WHERE fallback) -- same as v6
# =============================================================================
def compute_spatial_cas(eps_prompt, eps_null, eps_target, neighborhood_size=3):
    d_prompt = (eps_prompt - eps_null).float()
    d_target = (eps_target - eps_null).float()
    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2
    d_prompt_unfolded = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target, kernel_size=neighborhood_size, padding=pad)
    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)
    return spatial_cas.reshape(H, W)


def compute_soft_mask(spatial_cas, spatial_threshold=0.3, sigmoid_alpha=10.0,
                      blur_sigma=1.0, device=None):
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)
    return soft_mask.clamp(0, 1)


def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_h = g.view(1, 1, kernel_size, 1)
    kernel_w = g.view(1, 1, 1, kernel_size)
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2
    x = F.pad(x, [0, 0, pad_h, pad_h], mode='reflect')
    x = F.conv2d(x, kernel_h.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [pad_w, pad_w, 0, 0], mode='reflect')
    x = F.conv2d(x, kernel_w.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    return x


# =============================================================================
# Guidance Application (HOW) -- same as v6/v13
# =============================================================================
def apply_guidance(eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
                   soft_mask, guide_mode="hybrid", safety_scale=1.0,
                   cfg_scale=7.5, **kwargs):
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "sld":
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_null)

    elif guide_mode == "hybrid":
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * (eps_target - eps_null) \
                    + a_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "hybrid_proj":
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        p_scale = kwargs.get("proj_scale", 1.0)
        d_safe = d_prompt - p_scale * proj
        eps_safe_cfg = eps_null + cfg_scale * d_safe
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_safe_cfg * (1.0 - a_scale * mask) + eps_anchor_cfg * (a_scale * mask)

    elif guide_mode == "dag_adaptive":
        # DAG-inspired: area-based + magnitude-based adaptive scaling
        d_target_dir = eps_target - eps_null
        binary_mask = (mask > 0.5).to(mask.dtype)
        area = binary_mask.sum() / binary_mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])
        area_factor = area_scale * area * mask.numel()
        mag_scale = 1.0 + 4.0 * mask
        correction = safety_scale * area_factor * mag_scale * binary_mask * d_target_dir
        eps_final = eps_cfg - correction

    elif guide_mode == "projection":
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        d_safe = d_prompt - safety_scale * proj
        eps_safe_cfg = eps_null + cfg_scale * d_safe
        eps_final = eps_cfg * (1.0 - mask) + eps_safe_cfg * mask

    else:
        raise ValueError(f"Unknown guide_mode: {guide_mode}")

    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
    return eps_final


# =============================================================================
# Utils -- same as v6/v13
# =============================================================================
def load_prompts(filepath):
    filepath = Path(filepath)
    if filepath.suffix == ".csv":
        prompts = []
        with open(filepath, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            prompt_col = None
            for col in ['sensitive prompt', 'adv_prompt', 'prompt', 'target_prompt',
                        'text', 'Prompt', 'Text']:
                if col in fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"No known prompt column in {fieldnames}")
            for row in reader:
                p = row[prompt_col].strip()
                if p:
                    prompts.append(p)
        return prompts
    else:
        return [line.strip() for line in open(filepath) if line.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_safe_filename(prompt, max_len=50):
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)
    return safe[:max_len].replace(' ', '_')


def encode_concepts(text_encoder, tokenizer, concepts, device):
    all_embeds = []
    for concept in concepts:
        inputs = tokenizer(
            concept, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    p = ArgumentParser(description="IP-Adapter Image Projection + CAS Safe Generation v17")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    # v17-specific: IP-Adapter projection
    p.add_argument("--clip_embeddings", type=str,
                    default="exemplars/sd14/clip_exemplar_embeddings.pt",
                    help="Path to pre-computed CLIP exemplar embeddings (.pt)")
    p.add_argument("--ip_adapter_path", type=str, default=None,
                    help="Path to IP-Adapter weights (.bin). "
                         "If None, auto-downloads h94/IP-Adapter from Hub.")
    p.add_argument("--ip_num_tokens", type=int, default=4,
                    help="Number of image tokens from IP-Adapter Resampler (4 or 16)")

    # probe_source: WHICH probe(s) to compute (orthogonal to where_mode)
    p.add_argument("--probe_source", type=str, default="both",
                    choices=["text", "image", "both"],
                    help="Probe source for spatial mask: "
                         "text (text keyword cross-attn probe), "
                         "image (IP-Adapter projected probe), "
                         "both (max of text + image probes) [default]")

    # WHERE mode: HOW to combine probe mask with noise CAS mask
    p.add_argument("--where_mode", type=str, default="probe_only",
                    choices=["probe_only", "fused", "noise_only",
                             "ip_probe", "text_only"],  # legacy aliases kept
                    help="WHERE mode: "
                         "probe_only (use probe_source mask only), "
                         "fused (max of probe_source mask + noise CAS mask), "
                         "noise_only (noise CAS mask only, ignores probe_source). "
                         "Legacy aliases: ip_probe->probe_only, text_only->probe_only+text")

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE spatial params
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--target_words", type=str, nargs="+",
                    default=["nude", "naked", "nudity", "nsfw", "bare", "body"])
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["anchor_inpaint", "sld", "hybrid", "projection", "hybrid_proj", "dag_adaptive"])
    p.add_argument("--safety_scale", type=float, default=1.5)
    p.add_argument("--target_scale", type=float, default=-1.0)
    p.add_argument("--anchor_scale", type=float, default=-1.0)
    p.add_argument("--proj_scale", type=float, default=1.0)
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # Concepts (for CAS WHEN + text fallback)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"])

    # Multi-concept
    p.add_argument("--concept_packs", type=str, nargs="+", default=None,
                    help="Concept pack directories for multi-concept erasing. "
                         "If set, overrides --target_concepts, --anchor_concepts, etc.")
    p.add_argument("--family_level", action="store_true", default=False,
                    help="Apply corrections at family level (per sub-concept) instead of concept level")

    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False

    # Normalize legacy where_mode aliases
    if args.where_mode == "ip_probe":
        args.where_mode = "probe_only"
        if args.probe_source == "both":  # keep explicit probe_source if user set it
            args.probe_source = "image"  # ip_probe implied image only
    elif args.where_mode == "text_only":
        args.where_mode = "probe_only"
        args.probe_source = "text"

    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_concept = args.concept_packs is not None

    print(f"\n{'='*70}")
    print(f"IP-Adapter Image Projection + CAS Safe Generation v17")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: probe_source={args.probe_source}, where_mode={args.where_mode}, "
          f"ip_num_tokens={args.ip_num_tokens}, resolutions={args.attn_resolutions}")
    print(f"         spatial_threshold={args.spatial_threshold}, "
          f"sigmoid_alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  CLIP embeddings: {args.clip_embeddings}")
    print(f"  IP-Adapter: {args.ip_adapter_path or 'auto (h94/IP-Adapter hub)'}")
    print(f"  Model: {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # ---- Load prompts ----
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # ---- Load main pipeline ----
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    cross_attn_dim = 768  # SD1.4 cross-attention dimension

    # ---- Load IP-Adapter image projection ----
    image_proj_model = None
    fallback_mlp = None
    projection_mode = "none"

    need_image_probe = args.probe_source in ("image", "both") and args.where_mode != "noise_only"

    if need_image_probe:
        image_proj_model = load_ip_adapter_image_proj(
            ip_adapter_path=args.ip_adapter_path,
            device=device,
            dtype=torch.float32,
        )
        if image_proj_model is not None:
            projection_mode = "ip_adapter"
            print(f"  [v17] Using IP-Adapter Resampler projection")
        else:
            # Fallback to SimpleMLP
            print(f"  [v17] IP-Adapter failed, using SimpleMLP fallback (v17_simple)")
            fallback_mlp = SimpleMLP(
                clip_dim=768,
                cross_attn_dim=cross_attn_dim,
                num_tokens=args.ip_num_tokens,
                hidden_dim=1024,
            ).to(device).eval()
            for param in fallback_mlp.parameters():
                param.requires_grad_(False)
            projection_mode = "simple_mlp"

    # ---- Load CLIP exemplar embeddings ----
    clip_data = None
    ip_target_embeds = None
    ip_anchor_embeds = None

    if need_image_probe:
        clip_data = torch.load(args.clip_embeddings, map_location=device)

        # Get raw CLIP features for projection
        # clip_exemplar_embeddings.pt stores projected [1, 77, 768] embeddings
        # We need raw CLIP features for IP-Adapter projection
        raw_target_feat = clip_data.get("target_raw_clip", None)
        raw_anchor_feat = clip_data.get("anchor_raw_clip", None)

        if raw_target_feat is not None and projection_mode != "none":
            # Project raw CLIP features through IP-Adapter
            ip_target_embeds = project_clip_to_crossattn(
                raw_target_feat.to(device),
                image_proj_model, fallback_mlp,
                num_tokens=args.ip_num_tokens,
                cross_attn_dim=cross_attn_dim,
            )
            print(f"  [v17] IP-projected target embeds: {ip_target_embeds.shape}")
        else:
            # Use pre-computed CLIP embeddings from the .pt file directly
            # These are already in [1, 77, 768] format (from prepare_clip_exemplar.py)
            ip_target_embeds = clip_data["target_clip_embeds"].to(device=device, dtype=unet.dtype)
            print(f"  [v17] Using pre-computed CLIP target embeds: {ip_target_embeds.shape}")

        if raw_anchor_feat is not None and projection_mode != "none":
            ip_anchor_embeds = project_clip_to_crossattn(
                raw_anchor_feat.to(device),
                image_proj_model, fallback_mlp,
                num_tokens=args.ip_num_tokens,
                cross_attn_dim=cross_attn_dim,
            )
        else:
            ip_anchor_embeds = clip_data.get(
                "anchor_clip_embeds",
                clip_data["target_clip_embeds"],
            ).to(device)

        n_tokens_in_file = clip_data.get("config", {}).get("n_tokens", 4)
        print(f"  [v17] Projection mode: {projection_mode}, "
              f"file n_tokens={n_tokens_in_file}")

    # ---- Pre-encode text concept embeddings ----
    with torch.no_grad():
        target_text_embeds = encode_concepts(text_encoder, tokenizer,
                                             args.target_concepts, device)
        anchor_text_embeds = encode_concepts(text_encoder, tokenizer,
                                             args.anchor_concepts, device)
        uncond_inputs = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ==================================================================
    # Multi-concept vs single-concept setup
    # ==================================================================
    concept_packs_list = None
    pack_cas_gates = None
    pack_target_embeds_list = None
    pack_anchor_embeds_list = None
    pack_probe_token_indices = None

    # Single-concept state
    probe_store = AttentionProbeStore()
    probe_store_text = AttentionProbeStore()
    ip_probe_token_indices = None
    text_probe_token_indices = None
    original_processors = None

    if multi_concept:
        # ---- Multi-concept: load packs ----
        print(f"\nLoading {len(args.concept_packs)} concept packs...")
        concept_packs_list = load_multiple_packs(args.concept_packs, device=device)

        pack_cas_gates = []
        pack_target_embeds_list = []
        pack_anchor_embeds_list = []
        pack_probe_token_indices = []

        for i, pack in enumerate(concept_packs_list):
            pack_threshold = pack.get("cas_threshold", args.cas_threshold)
            pack_cas_gates.append(GlobalCAS(threshold=pack_threshold, sticky=args.cas_sticky))

            pack_target_concepts = pack.get("target_concepts", args.target_concepts)
            with torch.no_grad():
                t_emb = encode_concepts(text_encoder, tokenizer, pack_target_concepts, device)
            pack_target_embeds_list.append(t_emb)

            pack_anchor_concepts = pack.get("anchor_concepts", args.anchor_concepts)
            with torch.no_grad():
                a_emb = encode_concepts(text_encoder, tokenizer, pack_anchor_concepts, device)
            pack_anchor_embeds_list.append(a_emb)

            pack_target_words = pack.get("target_words", args.target_words)
            target_text = ", ".join(pack_target_concepts)
            p_indices = find_token_indices(target_text, pack_target_words, tokenizer)
            pack_probe_token_indices.append(p_indices)

            print(f"  Pack [{i}] '{pack.get('concept', 'unknown')}': "
                  f"threshold={pack_threshold}, "
                  f"targets={pack_target_concepts[:3]}{'...' if len(pack_target_concepts) > 3 else ''}, "
                  f"probe_tokens={p_indices[:6]}{'...' if len(p_indices) > 6 else ''}")

        # Shared probe store with combined target words
        combined_target_words = get_combined_target_words(concept_packs_list)
        combined_target_concepts = []
        for pack in concept_packs_list:
            combined_target_concepts.extend(pack.get("target_concepts", []))
        if combined_target_concepts:
            with torch.no_grad():
                combined_embeds = encode_concepts(
                    text_encoder, tokenizer, combined_target_concepts, device
                )
            combined_keys = precompute_target_keys(
                unet, combined_embeds.to(dtype=next(unet.parameters()).dtype),
                args.attn_resolutions
            )
            original_processors = register_attention_probe(
                unet, probe_store, combined_keys, args.attn_resolutions
            )
            combined_text = ", ".join(combined_target_concepts)
            probe_token_indices_combined = find_token_indices(
                combined_text, combined_target_words, tokenizer
            )
            print(f"  Multi-concept probe (text): combined_tokens={probe_token_indices_combined[:8]}"
                  f"{'...' if len(probe_token_indices_combined) > 8 else ''}")

        cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    else:
        # ---- Single-concept setup (original behavior) ----

        need_text_probe = args.probe_source in ("text", "both") and args.where_mode != "noise_only"

        if need_image_probe and need_text_probe:
            ip_target_keys = precompute_target_keys(
                unet, ip_target_embeds, args.attn_resolutions
            )
            ip_probe_token_indices = list(range(1, 1 + args.ip_num_tokens))
            text_target_keys = precompute_target_keys(
                unet, target_text_embeds, args.attn_resolutions
            )
            target_text = ", ".join(args.target_concepts)
            text_probe_token_indices = find_token_indices(
                target_text, args.target_words, tokenizer
            )
            original_processors = register_dual_attention_probe(
                unet, probe_store, probe_store_text,
                ip_target_keys, text_target_keys, args.attn_resolutions
            )
            print(f"  [v17] Both probes active (dual): IP token_indices={ip_probe_token_indices}, "
                  f"text token_indices={text_probe_token_indices}")

        elif need_image_probe:
            target_keys = precompute_target_keys(
                unet, ip_target_embeds, args.attn_resolutions
            )
            original_processors = register_attention_probe(
                unet, probe_store, target_keys, args.attn_resolutions
            )
            ip_probe_token_indices = list(range(1, 1 + args.ip_num_tokens))
            print(f"  [v17] IP probe token_indices={ip_probe_token_indices}")

        elif need_text_probe:
            target_keys = precompute_target_keys(
                unet, target_text_embeds, args.attn_resolutions
            )
            original_processors = register_attention_probe(
                unet, probe_store, target_keys, args.attn_resolutions
            )
            target_text = ", ".join(args.target_concepts)
            text_probe_token_indices = find_token_indices(
                target_text, args.target_words, tokenizer
            )
            print(f"  [v17] Text probe token_indices={text_probe_token_indices}")

        # Init CAS
        cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    # ---- Output directory ----
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

    # ---- Generation loop ----
    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        if not prompt.strip():
            continue

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(seed)
            cas.reset()

            guided_count = 0
            cas_values = []
            mask_areas = []

            # Encode the prompt
            with torch.no_grad():
                prompt_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

            # Init latents
            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)

            guide_start_step = int(total_steps * args.guide_start_frac)

            # ---- Denoising loop ----
            for step_idx, t in enumerate(scheduler.timesteps):

                lat_in = scheduler.scale_model_input(latents, t)

                # ==== Pass 1+2: CFG (with probe active) ====
                probe_store.active = True
                probe_store.reset()
                probe_store_text.active = True
                probe_store_text.reset()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                probe_store.active = False
                probe_store_text.active = False

                # ==== Pass 3: target concept (for CAS WHEN) ====
                with torch.no_grad():
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_text_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # =========================================================
                # WHEN/WHERE/HOW — Multi-concept or single-concept
                # =========================================================
                if multi_concept:
                    # ===================================================
                    # MULTI-CONCEPT: iterate over packs, accumulate corrections
                    # ===================================================
                    any_triggered = False
                    total_correction = torch.zeros_like(eps_cfg)
                    max_cas_val = 0.0

                    for pack_idx, pack in enumerate(concept_packs_list):
                        pack_cas_gate = pack_cas_gates[pack_idx]
                        p_target_embeds = pack_target_embeds_list[pack_idx]
                        p_anchor_embeds = pack_anchor_embeds_list[pack_idx]
                        p_probe_indices = pack_probe_token_indices[pack_idx]

                        p_guide_mode = pack.get("guide_mode", args.guide_mode)
                        p_safety_scale = pack.get("safety_scale", args.safety_scale)

                        # Target direction for this pack (text mode)
                        with torch.no_grad():
                            eps_target_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_target_embeds).sample

                        cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                            eps_prompt_pred, eps_null, eps_target_pack
                        )
                        max_cas_val = max(max_cas_val, cas_val_pack)

                        in_window = step_idx >= guide_start_step
                        if not (should_trigger_pack and in_window):
                            continue

                        any_triggered = True

                        # Anchor: compute online
                        with torch.no_grad():
                            eps_anchor_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_anchor_embeds).sample

                        # WHERE: compute per-pack mask from cross-attention probe
                        attn_spatial_pack = compute_attention_spatial_mask(
                            probe_store,
                            token_indices=p_probe_indices,
                            target_resolution=64,
                            resolutions_to_use=args.attn_resolutions,
                        )
                        soft_mask_pack = compute_soft_mask(
                            attn_spatial_pack.to(device),
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )

                        # Optional noise CAS fusion
                        if args.where_mode in ("fused", "noise_only"):
                            noise_spatial = compute_spatial_cas(
                                eps_prompt_pred, eps_null, eps_target_pack,
                                neighborhood_size=args.neighborhood_size,
                            )
                            noise_mask = compute_soft_mask(
                                noise_spatial,
                                spatial_threshold=args.spatial_threshold,
                                sigmoid_alpha=args.sigmoid_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )
                            if args.where_mode == "noise_only":
                                soft_mask_pack = noise_mask
                            else:
                                soft_mask_pack = torch.max(soft_mask_pack, noise_mask)

                        # HOW: compute per-pack correction
                        eps_guided_pack = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt_pred,
                            eps_target=eps_target_pack,
                            eps_anchor=eps_anchor_pack,
                            soft_mask=soft_mask_pack,
                            guide_mode=p_guide_mode,
                            safety_scale=p_safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=args.target_scale if args.target_scale > 0 else p_safety_scale,
                            anchor_scale=args.anchor_scale if args.anchor_scale > 0 else p_safety_scale,
                            proj_scale=args.proj_scale,
                        )
                        pack_correction = eps_cfg - eps_guided_pack
                        total_correction = total_correction + pack_correction

                        mask_areas.append(float(soft_mask_pack.mean().item()))

                    cas_val = max_cas_val
                    cas_values.append(cas_val)

                    if any_triggered:
                        eps_final = eps_cfg - total_correction
                        if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
                            eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
                        guided_count += 1
                    else:
                        eps_final = eps_cfg

                else:
                    # ===================================================
                    # SINGLE-CONCEPT: original v17 behavior
                    # ===================================================

                    # ==== Global CAS (WHEN) ====
                    cas_val, should_trigger = cas.compute(
                        eps_prompt_pred, eps_null, eps_target
                    )
                    cas_values.append(cas_val)

                    in_window = step_idx >= guide_start_step
                    should_guide = should_trigger and in_window

                    if should_guide:
                        # ==== Pass 4: anchor concept ====
                        with torch.no_grad():
                            eps_anchor = unet(lat_in, t,
                                              encoder_hidden_states=anchor_text_embeds).sample

                        # ==== WHERE: Spatial mask ====
                        probe_mask = None
                        attn_spatial = None

                        if args.where_mode != "noise_only":
                            if need_image_probe and need_text_probe:
                                ip_attn = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=ip_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                text_attn = compute_attention_spatial_mask(
                                    probe_store_text,
                                    token_indices=text_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                ip_soft = compute_soft_mask(
                                    ip_attn.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                                text_soft = compute_soft_mask(
                                    text_attn.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                                probe_mask = torch.max(ip_soft, text_soft)
                                attn_spatial = torch.max(ip_attn, text_attn)

                            elif need_image_probe:
                                attn_spatial = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=ip_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                probe_mask = compute_soft_mask(
                                    attn_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )

                            elif need_text_probe:
                                attn_spatial = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=text_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                probe_mask = compute_soft_mask(
                                    attn_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )

                        noise_mask = None
                        if args.where_mode in ("fused", "noise_only"):
                            noise_spatial = compute_spatial_cas(
                                eps_prompt_pred, eps_null, eps_target,
                                neighborhood_size=args.neighborhood_size,
                            )
                            noise_mask = compute_soft_mask(
                                noise_spatial,
                                spatial_threshold=args.spatial_threshold,
                                sigmoid_alpha=args.sigmoid_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )
                            if attn_spatial is None:
                                attn_spatial = noise_spatial

                        if args.where_mode == "probe_only":
                            soft_mask = probe_mask
                        elif args.where_mode == "fused":
                            soft_mask = torch.max(probe_mask, noise_mask)
                        elif args.where_mode == "noise_only":
                            soft_mask = noise_mask

                        # ==== HOW: Apply guidance ====
                        eps_final = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt_pred,
                            eps_target=eps_target,
                            eps_anchor=eps_anchor,
                            soft_mask=soft_mask,
                            guide_mode=args.guide_mode,
                            safety_scale=args.safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=args.target_scale if args.target_scale > 0 else args.safety_scale,
                            anchor_scale=args.anchor_scale if args.anchor_scale > 0 else args.safety_scale,
                            proj_scale=args.proj_scale,
                        )

                        guided_count += 1
                        area_val = float(soft_mask.mean().item())
                        mask_areas.append(area_val)

                        # Save maps
                        if args.save_maps and step_idx % 10 == 0:
                            attn_map_np = attn_spatial.float().cpu().numpy()
                            attn_map_np = np.nan_to_num(attn_map_np, nan=0.0)
                            attn_map_img = (np.clip(attn_map_np, 0, 1) * 255).astype(np.uint8)
                            Image.fromarray(attn_map_img, 'L').save(
                                str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_attn.png"))
                            mask_np = soft_mask[0, 0].float().cpu().numpy()
                            mask_img = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                            Image.fromarray(mask_img, 'L').save(
                                str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_mask.png"))

                    else:
                        eps_final = eps_cfg

                # ==== DDIM step ====
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}")

            # ---- Decode ----
            with torch.no_grad():
                decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            slug = make_safe_filename(prompt)
            fname = f"{prompt_idx:04d}_{sample_idx:02d}_{slug}.png"
            Image.fromarray(img_np).resize((512, 512)).save(str(outdir / fname))

            stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": seed,
                "prompt": prompt[:100],
                "filename": fname,
                "guided_steps": guided_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_count / max(total_steps, 1),
                "cas_triggered": cas.triggered,
                "avg_cas": float(np.mean(cas_values)) if cas_values else 0.0,
                "max_cas": float(np.max(cas_values)) if cas_values else 0.0,
                "avg_mask_area": float(np.mean(mask_areas)) if mask_areas else 0.0,
                "max_mask_area": float(np.max(mask_areas)) if mask_areas else 0.0,
                "probe_source": args.probe_source,
                "where_mode": args.where_mode,
                "projection_mode": projection_mode,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f} "
                    f"probe={args.probe_source} where={args.where_mode} "
                    f"proj={projection_mode}"
                )

    # ---- Cleanup ----
    if original_processors is not None:
        restore_processors(unet, original_processors)

    # ---- Save stats ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "IP-Adapter Image Projection + CAS Safe Generation v17",
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "trigger_rate": n_trig / max(n, 1),
            "avg_guided_steps": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
            "avg_mask_area": float(np.mean([s["avg_mask_area"] for s in all_stats if s["avg_mask_area"] > 0])) if n else 0,
            "projection_mode": projection_mode,
        },
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"  probe_source: {args.probe_source}, where_mode: {args.where_mode}, "
          f"projection: {projection_mode}")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
