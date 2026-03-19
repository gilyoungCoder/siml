#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Unlearning with LEAVE-ONE-OUT (LOO) Token Importance

Key Innovation:
  - Instead of simple cosine similarity, we measure token CRITICALITY
  - LOO approach: Remove each token and measure embedding change
  - Critical tokens (e.g., "nude") cause large embedding shifts when removed
  - Non-critical tokens (e.g., "a", "the") cause small changes
  - Combine criticality with cosine similarity for smarter suppression

Algorithm:
  1. For each token i in prompt:
     - Compute full embedding E_full
     - Compute embedding without token i: E_-i
     - Criticality_i = ||E_full - E_-i|| (L2 distance)
  2. Suppression score = criticality × cosine_similarity
  3. Suppress tokens with high combined score

Features:
  - Adaptive threshold based on criticality distribution
  - Per-token criticality visualization
  - Smarter suppression of truly harmful tokens
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

import numpy as np
from typing import List, Optional, Tuple, Dict


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Machine Unlearning with Leave-One-Out Token Importance")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="output_img/unlearning_loo", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Harmful concept suppression with LOO
    parser.add_argument("--harm_suppress", action="store_true", help="Enable harmful concept suppression")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt",
                        help="File containing harmful concepts to suppress (one per line)")

    # LOO Parameters
    parser.add_argument("--loo_mode", type=str, default="combined", choices=["cosine", "criticality", "combined"],
                        help="Suppression mode: cosine (original), criticality (LOO only), combined (both)")
    parser.add_argument("--criticality_weight", type=float, default=1.0,
                        help="Weight for criticality in combined mode (score = cosine * criticality^weight)")
    parser.add_argument("--criticality_normalize", action="store_true",
                        help="Normalize criticality scores to [0, 1] range")

    # Adaptive threshold
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use adaptive threshold based on score distribution")
    parser.add_argument("--base_tau", type=float, default=0.15,
                        help="Base threshold")
    parser.add_argument("--central_percentile", type=float, default=0.80,
                        help="Central percentile for computing mean (default: 0.80 = 80%)")
    parser.add_argument("--tau_factor", type=float, default=1.05,
                        help="Multiplicative factor for adaptive threshold")

    parser.add_argument("--harm_gamma_start", type=float, default=40.0,
                        help="Suppression strength at early steps")
    parser.add_argument("--harm_gamma_end", type=float, default=0.5,
                        help="Suppression strength at late steps")

    # Classifier Guidance
    parser.add_argument("--classifier_guidance", action="store_true", help="Enable classifier guidance")
    parser.add_argument("--classifier_config", type=str,
                        default="./configs/models/time_dependent_discriminator.yaml",
                        help="Classifier model configuration file")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint path")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step to start applying guidance")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance (1=clothed people)")

    # DEBUG options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug_prompts", action="store_true", help="Show per-prompt token analysis")
    parser.add_argument("--debug_steps", action="store_true", help="Show per-step attention analysis")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def save_image(image, filename, root="output_img"):
    """Save generated image to disk."""
    path = os.path.join(root, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    image = np.asarray(image)
    image = Image.fromarray(image, mode="RGB")
    image = image.resize((512, 512))
    image.save(path)


def schedule_linear(step: int, num_steps: int, start_val: float, end_val: float) -> float:
    """Linear scheduling from start_val to end_val."""
    t = step / max(1, num_steps - 1)
    return start_val * (1.0 - t) + end_val * t


# =========================
# LEAVE-ONE-OUT Token Criticality
# =========================
@torch.no_grad()
def compute_token_criticality(
    pipe,
    prompt: str,
    content_indices: List[int]
) -> torch.Tensor:
    """
    Compute Leave-One-Out (LOO) criticality for each token.

    Criticality measures how much the embedding changes when a token is removed.
    High criticality = token is important to the prompt's meaning.

    Args:
        pipe: Diffusion pipeline with tokenizer and text encoder
        prompt: Input prompt string
        content_indices: List of content token indices (excluding special tokens)

    Returns:
        criticality_scores: (num_content_tokens,) tensor of criticality scores
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    # Tokenize full prompt
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Get full embedding
    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states_full = outputs.hidden_states[-2][0]  # (L, d)

    # Compute mean embedding of content tokens (this is what we care about)
    content_mask = torch.zeros(hidden_states_full.shape[0], dtype=torch.bool, device=device)
    content_mask[content_indices] = True

    # Full embedding (mean of content tokens)
    embedding_full = hidden_states_full[content_mask].mean(dim=0)  # (d,)

    # For each content token, compute embedding without it
    criticality_scores = []

    for idx in content_indices:
        # Create mask excluding current token
        loo_mask = content_mask.clone()
        loo_mask[idx] = False

        if loo_mask.sum() == 0:
            # If only one token, criticality is maximal
            criticality_scores.append(1.0)
            continue

        # Compute embedding without this token
        embedding_loo = hidden_states_full[loo_mask].mean(dim=0)  # (d,)

        # Criticality = L2 distance between full and LOO embeddings
        criticality = torch.norm(embedding_full - embedding_loo, p=2).item()
        criticality_scores.append(criticality)

    return torch.tensor(criticality_scores, device=device)


# =========================
# Enhanced Token Analysis with LOO
# =========================
@torch.no_grad()
def compute_loo_token_scores(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor,
    mode: str = "combined",
    criticality_weight: float = 1.0,
    criticality_normalize: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute token-level scores using LOO criticality.

    Returns:
        dict with keys:
            - cosine_sims: (L,) cosine similarities with harm vector
            - criticality: (num_content,) criticality scores
            - combined_scores: (L,) final suppression scores
            - content_mask: (L,) boolean mask for content tokens
            - token_strings: List of token strings
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    # Tokenize
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Get embeddings
    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-2][0]  # (L, d)
    hidden_states = F.normalize(hidden_states, dim=-1)

    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]

    # Build content mask
    content_mask = attention_mask.bool().clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    # Get content indices
    content_indices = torch.where(content_mask)[0].tolist()

    # Compute cosine similarities
    harm_vec_normalized = F.normalize(harm_vector.to(device), dim=-1)
    cosine_sims = torch.matmul(hidden_states, harm_vec_normalized)  # (L,)

    # Compute LOO criticality for content tokens
    if len(content_indices) > 0:
        criticality_scores = compute_token_criticality(pipe, prompt, content_indices)

        # Normalize criticality to [0, 1] if requested
        if criticality_normalize and criticality_scores.numel() > 0:
            min_crit = criticality_scores.min()
            max_crit = criticality_scores.max()
            if max_crit > min_crit:
                criticality_scores = (criticality_scores - min_crit) / (max_crit - min_crit)
    else:
        criticality_scores = torch.tensor([], device=device)

    # Compute combined scores for all tokens
    combined_scores = torch.zeros_like(cosine_sims)

    if mode == "cosine":
        # Original: only cosine similarity
        combined_scores = cosine_sims
    elif mode == "criticality":
        # Only criticality
        for i, idx in enumerate(content_indices):
            combined_scores[idx] = criticality_scores[i]
    elif mode == "combined":
        # Multiply cosine similarity by criticality
        for i, idx in enumerate(content_indices):
            # Score = cosine × (criticality ^ weight)
            # This gives higher scores to tokens that are both similar AND critical
            combined_scores[idx] = cosine_sims[idx] * (criticality_scores[i] ** criticality_weight)

    # Get token strings
    token_strings = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    return {
        'cosine_sims': cosine_sims,
        'criticality': criticality_scores,
        'combined_scores': combined_scores,
        'content_mask': content_mask,
        'token_strings': token_strings,
        'content_indices': content_indices,
    }


# =========================
# Adaptive Threshold
# =========================
@torch.no_grad()
def compute_adaptive_threshold(
    scores: torch.Tensor,
    central_percentile: float = 0.80,
    factor: float = 1.05,
    base_tau: float = 0.15
) -> float:
    """
    Compute adaptive threshold based on central mean of score distribution.
    """
    if scores.numel() == 0:
        return base_tau

    sorted_scores = torch.sort(scores)[0]
    n = len(sorted_scores)

    if n < 3:
        return base_tau

    # Calculate central region
    tail_fraction = (1.0 - central_percentile) / 2.0
    start_idx = int(n * tail_fraction)
    end_idx = int(n * (1.0 - tail_fraction))
    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)

    if end_idx <= start_idx:
        return base_tau

    # Compute central mean
    central_scores = sorted_scores[start_idx:end_idx]
    central_mean = float(central_scores.mean().item())

    # Adaptive threshold
    adaptive_tau = central_mean * factor
    adaptive_tau = max(0.0, min(1.0, adaptive_tau))

    return adaptive_tau


# =========================
# DEBUG: Token Analysis with LOO
# =========================
@torch.no_grad()
def debug_loo_token_analysis(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor,
    harm_concepts: List[str],
    tau: float,
    mode: str,
    criticality_weight: float,
    criticality_normalize: bool,
    adaptive_tau: Optional[float] = None,
):
    """
    DEBUG: Analyze each token with LOO criticality information.
    """
    print("\n" + "="*120)
    print(f"[DEBUG] LEAVE-ONE-OUT TOKEN ANALYSIS FOR PROMPT: '{prompt}'")
    print("="*120)

    # Compute scores
    result = compute_loo_token_scores(
        pipe, prompt, harm_vector,
        mode=mode,
        criticality_weight=criticality_weight,
        criticality_normalize=criticality_normalize
    )

    cosine_sims = result['cosine_sims']
    criticality = result['criticality']
    combined_scores = result['combined_scores']
    content_mask = result['content_mask']
    token_strings = result['token_strings']
    content_indices = result['content_indices']

    # Determine effective tau
    effective_tau = adaptive_tau if adaptive_tau is not None else tau

    # Print header
    print(f"\nHarm Concepts: {', '.join(harm_concepts)}")
    print(f"Mode: {mode.upper()}")
    if mode == "combined":
        print(f"  - Criticality weight: {criticality_weight}")
        print(f"  - Criticality normalization: {criticality_normalize}")

    if adaptive_tau is not None:
        print(f"\nThreshold Method: ADAPTIVE")
        print(f"  - Base τ: {tau:.3f}")
        print(f"  - Adaptive τ: {adaptive_tau:.3f} ⭐")
        content_scores = combined_scores[content_mask]
        if content_scores.numel() > 0:
            print(f"  - Score stats: min={content_scores.min():.3f}, "
                  f"mean={content_scores.mean():.3f}, max={content_scores.max():.3f}")
    else:
        print(f"\nThreshold Method: FIXED (τ = {tau:.3f})")

    print(f"\n{'IDX':<4} {'TOKEN':<20} {'TYPE':<10} {'COSINE':<10} {'CRITICALITY':<12} {'COMBINED':<10} {'≥ τ?':<8} {'SUPPRESS?':<10}")
    print("-"*120)

    suppressed_count = 0
    total_content_tokens = 0

    # Map content indices to criticality scores
    crit_map = {}
    for i, idx in enumerate(content_indices):
        crit_map[idx] = criticality[i].item() if i < len(criticality) else 0.0

    for idx in range(len(token_strings)):
        if idx >= len(cosine_sims):
            break

        token_str = token_strings[idx].replace('Ġ', '▁').replace('</w>', '')

        # Determine token type
        if not content_mask[idx]:
            token_type = "SPECIAL"
        else:
            token_type = "CONTENT"

        # Get scores
        cosine = cosine_sims[idx].item()
        crit = crit_map.get(idx, 0.0)
        score = combined_scores[idx].item()

        # Check if suppressed
        is_content = bool(content_mask[idx].item())
        exceeds_threshold = (score >= effective_tau)
        is_suppressed = is_content and exceeds_threshold

        threshold_str = "YES ⚠️" if exceeds_threshold else "NO"
        suppressed_str = "YES 🔴" if is_suppressed else "NO"

        print(f"{idx:<4} {token_str:<20} {token_type:<10} {cosine:+.4f}    {crit:.4f}      {score:+.4f}    {threshold_str:<8} {suppressed_str:<10}")

        if is_suppressed:
            suppressed_count += 1
        if is_content:
            total_content_tokens += 1

    print("-"*120)
    print(f"Summary: {suppressed_count}/{total_content_tokens} content tokens suppressed "
          f"({100*suppressed_count/max(1, total_content_tokens):.1f}%)")

    if adaptive_tau is not None:
        ratio = adaptive_tau / tau if tau > 0 else 1.0
        print(f"Adaptive Threshold Effect: τ changed from {tau:.3f} → {adaptive_tau:.3f} "
              f"(×{ratio:.3f}, Δ = {adaptive_tau - tau:+.3f})")

    print("="*120 + "\n")


# =========================
# Vector Building
# =========================
@torch.no_grad()
def build_harm_vector(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    """Build a normalized vector representing harmful concepts."""
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
# Attention Processor with LOO
# =========================
class LOOHarmSuppressionAttnProcessor(AttnProcessor2_0):
    """
    Attention processor with Leave-One-Out criticality support.
    """

    def __init__(self,
                 harm_vector: Optional[torch.Tensor] = None,
                 tau: float = 0.15,
                 gamma: float = 1.0,
                 debug: bool = False,
                 mode: str = "combined",
                 criticality_weight: float = 1.0):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.training = False
        self.debug = debug
        self.mode = mode
        self.criticality_weight = criticality_weight

        # Adaptive threshold
        self.current_adaptive_tau = None

        # DEBUG tracking
        self.debug_step_count = 0
        self.debug_suppression_stats = []

        # Store harm vector
        if harm_vector is None or harm_vector.numel() == 0:
            self._harm_vector = None
        else:
            self._harm_vector = F.normalize(harm_vector.detach().float(), dim=-1).cpu()

    @property
    def harm_vector(self) -> Optional[torch.Tensor]:
        return self._harm_vector

    def set_harm_vector(self, harm_vector: Optional[torch.Tensor]):
        if harm_vector is None or harm_vector.numel() == 0:
            self._harm_vector = None
        else:
            self._harm_vector = F.normalize(harm_vector.detach().float(), dim=-1).cpu()

    def set_gamma(self, gamma: float):
        self.gamma = float(gamma)

    def set_adaptive_tau(self, adaptive_tau: Optional[float]):
        self.current_adaptive_tau = adaptive_tau

    def get_effective_tau(self) -> float:
        return self.current_adaptive_tau if self.current_adaptive_tau is not None else self.tau

    def set_suppression_scores(self, scores: Optional[torch.Tensor], content_mask: Optional[torch.Tensor] = None):
        """Set pre-computed suppression scores for current prompt."""
        self._suppression_scores = scores
        self._content_mask = content_mask

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
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

        # Apply LOO-based suppression (cross-attention only)
        if is_cross_attn and self._harm_vector is not None and hasattr(self, '_suppression_scores'):
            device = scores.device
            suppression_scores = self._suppression_scores.to(device)  # (K,)

            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            num_heads = scores.shape[0] // B

            # Get effective threshold
            effective_tau = self.get_effective_tau()

            # Suppress tokens with score >= effective_tau, but ONLY content tokens
            suppress_mask = (suppression_scores >= effective_tau)

            # Apply content mask if available to only suppress content tokens
            if hasattr(self, '_content_mask') and self._content_mask is not None:
                content_mask_device = self._content_mask.to(device)
                suppress_mask = suppress_mask & content_mask_device

            # DEBUG
            if self.debug and is_cross_attn:
                num_suppressed = suppress_mask.sum().item()
                total_tokens = suppress_mask.numel()
                avg_score = suppression_scores.mean().item()
                max_score = suppression_scores.max().item()

                self.debug_suppression_stats.append({
                    'step': self.debug_step_count,
                    'gamma': self.gamma,
                    'tau': effective_tau,
                    'num_suppressed': num_suppressed,
                    'total_tokens': total_tokens,
                    'suppression_rate': num_suppressed / max(1, total_tokens),
                    'avg_score': avg_score,
                    'max_score': max_score,
                })

            if suppress_mask.any():
                weight = suppression_scores.clamp(min=0.0) * suppress_mask.float()
                weight_expanded = weight[None, None, :].expand(B, Q, K).repeat_interleave(num_heads, dim=0)
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
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def print_step_debug(self, step: int):
        """Print debug info for recent step."""
        if self.debug and self.debug_suppression_stats:
            recent = self.debug_suppression_stats[-1]
            print(f"  [DEBUG Step {step:02d}] γ={recent['gamma']:.2f} | τ={recent['tau']:.3f} | "
                  f"Suppressed: {recent['num_suppressed']}/{recent['total_tokens']} ({100*recent['suppression_rate']:.1f}%) | "
                  f"Score: avg={recent['avg_score']:+.4f}, max={recent['max_score']:+.4f}")


# =========================
# Main Generation Loop
# =========================
def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*100}")
    print(f"[INFO] Machine Unlearning - LEAVE-ONE-OUT (LOO) MODE")
    print(f"{'='*100}")
    print(f"[INFO] Loading model from {args.ckpt_path}")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        safety_checker=None
    ).to(device)
    print(f"[INFO] Model loaded on device: {pipe.device}")

    # Load prompts
    prompt_file = os.path.expanduser(args.prompt_file)
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(prompts)} prompts from {prompt_file}")

    # Prepare output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # Setup Harmful Concept Suppression with LOO
    # =========================
    harm_processor = None
    harm_concepts = []
    if args.harm_suppress:
        harm_file = os.path.expanduser(args.harm_concepts_file)
        if os.path.isfile(harm_file):
            with open(harm_file, "r") as f:
                harm_concepts = [line.strip() for line in f if line.strip()]

            print(f"\n[INFO] Building harmful concept vector from {len(harm_concepts)} concepts: {harm_concepts}")
            harm_vector = build_harm_vector(pipe, harm_concepts)

            harm_processor = LOOHarmSuppressionAttnProcessor(
                harm_vector=harm_vector,
                tau=args.base_tau,
                gamma=args.harm_gamma_start,
                debug=args.debug or args.debug_steps,
                mode=args.loo_mode,
                criticality_weight=args.criticality_weight
            )
            pipe.unet.set_attn_processor(harm_processor)

            print(f"[INFO] LOO-based harmful concept suppression enabled")
            print(f"  - Mode: {args.loo_mode.upper()}")
            if args.loo_mode == "combined":
                print(f"  - Criticality weight: {args.criticality_weight}")
                print(f"  - Criticality normalization: {args.criticality_normalize}")

            if args.adaptive_threshold:
                print(f"  - Threshold mode: ADAPTIVE ⭐")
                print(f"  - Base threshold: {args.base_tau}")
                print(f"  - Central percentile: {args.central_percentile*100:.0f}%")
                print(f"  - Multiplicative factor: ×{args.tau_factor}")
            else:
                print(f"  - Threshold mode: FIXED (τ = {args.base_tau})")
            print(f"  - Gamma schedule: {args.harm_gamma_start} → {args.harm_gamma_end}")
        else:
            print(f"[WARNING] Harmful concepts file not found: {harm_file}")
            args.harm_suppress = False

    # =========================
    # Setup Classifier Guidance
    # =========================
    guidance_model = None
    if args.classifier_guidance:
        print(f"\n[INFO] Loading classifier from {args.classifier_ckpt}")
        guidance_model = GuidanceModel(
            pipe,
            args.classifier_config,
            args.classifier_ckpt,
            1,
            device
        )
        print(f"[INFO] Classifier guidance enabled")
        print(f"  - Scale: {args.guidance_scale}")
        print(f"  - Target class: {args.target_class} (clothed people)")
        print(f"  - Start step: {args.guidance_start_step}")

    # =========================
    # Generation Loop
    # =========================
    def callback_on_step_end(
        diffusion_pipeline,
        step,
        timestep,
        callback_kwargs,
        guidance_model,
        guidance_scale,
        guidance_start_step,
        target_class,
        harm_processor,
        num_steps,
    ):
        """Callback function called at the end of each denoising step."""

        # Update suppression strength (gamma) schedule
        if harm_processor is not None:
            # Clamp step to valid range [0, num_steps-1] to prevent negative gamma
            step_clamped = min(step, num_steps - 1)
            gamma = schedule_linear(step_clamped, num_steps, args.harm_gamma_start, args.harm_gamma_end)
            harm_processor.set_gamma(gamma)
            harm_processor.debug_step_count = step

            # DEBUG: Print step info
            if args.debug_steps:
                harm_processor.print_step_debug(step)

        # Apply classifier guidance
        if guidance_model is not None and step >= guidance_start_step:
            callback_kwargs = guidance_model.guidance(
                diffusion_pipeline,
                callback_kwargs,
                step,
                timestep,
                guidance_scale,
                target_class=target_class
            )

        return callback_kwargs

    # Generate images for each prompt
    print(f"\n{'='*100}")
    print(f"[INFO] Starting generation...")
    print(f"{'='*100}\n")

    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*100}")
        print(f"[PROMPT {idx + 1}/{len(prompts)}] {prompt}")
        print(f"{'='*100}")

        # LOO: Compute token scores for this prompt
        adaptive_tau = None

        if harm_processor is not None and harm_processor.harm_vector is not None:
            # Compute LOO-based scores
            result = compute_loo_token_scores(
                pipe, prompt, harm_processor.harm_vector,
                mode=args.loo_mode,
                criticality_weight=args.criticality_weight,
                criticality_normalize=args.criticality_normalize
            )

            combined_scores = result['combined_scores']
            content_mask = result['content_mask']

            # Store scores AND content mask in processor for use during attention
            harm_processor.set_suppression_scores(combined_scores, content_mask)

            # Compute adaptive threshold if enabled
            if args.adaptive_threshold:
                content_scores = combined_scores[content_mask]

                if content_scores.numel() > 0:
                    adaptive_tau = compute_adaptive_threshold(
                        content_scores,
                        central_percentile=args.central_percentile,
                        factor=args.tau_factor,
                        base_tau=args.base_tau
                    )
                    harm_processor.set_adaptive_tau(adaptive_tau)

                    print(f"\n[ADAPTIVE THRESHOLD]")
                    print(f"  Base τ: {args.base_tau:.3f}")
                    print(f"  → Adaptive τ: {adaptive_tau:.3f} ⭐")
                else:
                    harm_processor.set_adaptive_tau(args.base_tau)
                    adaptive_tau = args.base_tau
            else:
                harm_processor.set_adaptive_tau(args.base_tau)

        # DEBUG: Analyze prompt tokens with LOO
        if args.debug or args.debug_prompts:
            if harm_processor is not None and harm_processor.harm_vector is not None:
                debug_loo_token_analysis(
                    pipe,
                    prompt,
                    harm_processor.harm_vector,
                    harm_concepts,
                    args.base_tau,
                    args.loo_mode,
                    args.criticality_weight,
                    args.criticality_normalize,
                    adaptive_tau=adaptive_tau,
                )

        # Prepare callback
        if args.harm_suppress or args.classifier_guidance:
            callback = partial(
                callback_on_step_end,
                guidance_model=guidance_model if args.classifier_guidance else None,
                guidance_scale=args.guidance_scale,
                guidance_start_step=args.guidance_start_step,
                target_class=args.target_class,
                harm_processor=harm_processor,
                num_steps=args.num_inference_steps,
            )
            callback_tensor_inputs = ["latents", "noise_pred", "prev_latents"]
            if args.classifier_guidance:
                callback_tensor_inputs += ["instance_prompt_embeds"]
        else:
            callback = None
            callback_tensor_inputs = None

        # Reset debug stats AND gamma for new prompt
        if harm_processor is not None:
            harm_processor.debug_suppression_stats = []
            harm_processor.debug_step_count = 0
            # CRITICAL: Reset gamma to start value for new prompt
            harm_processor.set_gamma(args.harm_gamma_start)

        # Generate
        print(f"\n[INFO] Generating {args.nsamples} image(s)...")
        with torch.enable_grad():
            output = pipe(
                prompt=prompt,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                height=512,
                width=512,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=callback_tensor_inputs,
                num_images_per_prompt=args.nsamples,
            )

        # DEBUG: Print generation statistics
        if args.debug and harm_processor is not None and harm_processor.debug_suppression_stats:
            print(f"\n[DEBUG] Generation Statistics:")
            stats = harm_processor.debug_suppression_stats
            avg_rate = np.mean([s['suppression_rate'] for s in stats])
            avg_score = np.mean([s['avg_score'] for s in stats])
            max_score_overall = max([s['max_score'] for s in stats])
            avg_tau = np.mean([s['tau'] for s in stats])
            print(f"  - Average suppression rate: {100*avg_rate:.1f}%")
            print(f"  - Average score: {avg_score:+.4f}")
            print(f"  - Max score: {max_score_overall:+.4f}")
            print(f"  - Average threshold used: {avg_tau:.3f}")

        # Save images
        for sample_idx, image in enumerate(output.images):
            filename = f"prompt_{idx+1:04d}_sample_{sample_idx+1}.png"
            save_image(image, filename, root=output_dir)

        print(f"\n[INFO] Saved {len(output.images)} image(s) to {output_dir}")

    print(f"\n{'='*100}")
    print(f"[INFO] Generation complete! All images saved to {output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
