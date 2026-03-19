#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Unlearning with ADAPTIVE THRESHOLDING - Enhanced Implementation

Features:
  1. Attention Manipulation: Suppressing harmful concepts via cosine similarity
  2. ADAPTIVE THRESHOLD: Dynamic threshold calculation per prompt (SAFREE-style)
  3. Classifier Guidance: Guiding generation towards "clothed people" class
  4. DEBUG: Per-token analysis with cosine similarities and dynamic threshold

Key Innovation:
  - Fixed threshold → Adaptive threshold based on central mean (80%)
  - Robust to prompt length and ambiguous words
  - Prompt-adaptive suppression strategy
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
from typing import List, Optional, Tuple


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Machine Unlearning with ADAPTIVE THRESHOLD + DEBUG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="output_img/unlearning_adaptive", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Harmful concept suppression (Attention Manipulation)
    parser.add_argument("--harm_suppress", action="store_true", help="Enable harmful concept suppression")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt",
                        help="File containing harmful concepts to suppress (one per line)")

    # ADAPTIVE THRESHOLD parameters
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use adaptive threshold based on prompt distribution")
    parser.add_argument("--base_tau", type=float, default=0.15,
                        help="Base threshold (used if adaptive disabled, or as fallback)")
    parser.add_argument("--central_percentile", type=float, default=0.80,
                        help="Central percentile for computing mean (default: 0.80 = 80%)")
    parser.add_argument("--tau_factor", type=float, default=1.02,
                        help="Multiplicative factor for adaptive threshold: tau = central_mean * factor")

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
# ADAPTIVE THRESHOLD CALCULATION
# =========================
@torch.no_grad()
def compute_adaptive_threshold(
    cosine_similarities: torch.Tensor,
    central_percentile: float = 0.80,
    factor: float = 1.02,
    base_tau: float = 0.15
) -> float:
    """
    Compute adaptive threshold based on central mean of cosine similarity distribution.

    Uses MULTIPLICATIVE factor for scale-invariant thresholding.

    Args:
        cosine_similarities: (N,) tensor of cosine similarities
        central_percentile: Fraction of central data to use (default: 0.80 = 80%)
        factor: Multiplicative factor (default: 1.02 = 2% above central mean)
        base_tau: Fallback threshold if computation fails

    Returns:
        Adaptive threshold value

    Example:
        If similarities = [0.01, 0.02, 0.03, ..., 0.15, ..., 0.89]
        And central_percentile = 0.80, factor = 1.02:
        1. Sort similarities
        2. Remove top 10% and bottom 10%
        3. Compute mean of remaining 80%
        4. tau = central_mean × 1.02 (2% above mean)
    """
    if cosine_similarities.numel() == 0:
        return base_tau

    # Sort similarities
    sorted_sims = torch.sort(cosine_similarities)[0]
    n = len(sorted_sims)

    if n < 3:
        # Too few samples, use base threshold
        return base_tau

    # Calculate central region indices
    tail_fraction = (1.0 - central_percentile) / 2.0
    start_idx = int(n * tail_fraction)
    end_idx = int(n * (1.0 - tail_fraction))

    # Ensure valid range
    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)

    if end_idx <= start_idx:
        return base_tau

    # Compute central mean
    central_sims = sorted_sims[start_idx:end_idx]
    central_mean = float(central_sims.mean().item())

    # Adaptive threshold = central mean × factor (MULTIPLICATIVE)
    adaptive_tau = central_mean * factor

    # Clamp to reasonable range [0.0, 1.0]
    adaptive_tau = max(0.0, min(1.0, adaptive_tau))

    return adaptive_tau


# =========================
# DEBUG: Token Analysis
# =========================
@torch.no_grad()
def debug_token_analysis(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor,
    harm_concepts: List[str],
    tau: float,
    adaptive_tau: Optional[float] = None,
    tau_method: str = "fixed"
):
    """
    DEBUG: Analyze each token with adaptive threshold information.
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    print("\n" + "="*100)
    print(f"[DEBUG] TOKEN ANALYSIS FOR PROMPT: '{prompt}'")
    print("="*100)

    # Tokenize prompt
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

    # Get token strings
    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    # Normalize harm vector
    harm_vec_normalized = F.normalize(harm_vector.to(device), dim=-1)

    # Build content mask
    content_mask = attention_mask.bool().clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    # Compute cosine similarities
    cosine_sims = []
    for idx in range(len(input_ids)):
        if not attention_mask[idx]:
            break
        sim = float(torch.dot(hidden_states[idx], harm_vec_normalized).item())
        cosine_sims.append(sim)

    cosine_sims_tensor = torch.tensor(cosine_sims, device=device)

    # Print header
    print(f"\nHarm Concepts: {', '.join(harm_concepts)}")

    if tau_method == "adaptive" and adaptive_tau is not None:
        print(f"Threshold Method: ADAPTIVE (Multiplicative)")
        print(f"  - Base τ: {tau:.3f}")
        print(f"  - Adaptive τ: {adaptive_tau:.3f} ⭐")
        print(f"  - Distribution stats: min={cosine_sims_tensor.min():.3f}, "
              f"mean={cosine_sims_tensor.mean():.3f}, max={cosine_sims_tensor.max():.3f}")
        effective_tau = adaptive_tau
    else:
        print(f"Threshold Method: FIXED")
        print(f"  - Fixed τ: {tau:.3f}")
        effective_tau = tau

    print(f"\n{'IDX':<4} {'TOKEN':<20} {'TYPE':<12} {'COSINE_SIM':<12} {'≥ τ?':<8} {'SUPPRESSED?':<12}")
    print("-"*100)

    suppressed_count = 0
    total_content_tokens = 0

    for idx in range(len(input_ids)):
        if not attention_mask[idx]:
            break

        token_str = token_strings[idx].replace('Ġ', '▁').replace('</w>', '')
        token_id = int(input_ids[idx].item())

        # Determine token type
        token_type = "CONTENT"
        if tokenizer.bos_token_id and token_id == tokenizer.bos_token_id:
            token_type = "BOS/SOT"
        elif tokenizer.eos_token_id and token_id == tokenizer.eos_token_id:
            token_type = "EOS/EOT"
        elif tokenizer.pad_token_id and token_id == tokenizer.pad_token_id:
            token_type = "PAD"

        # Compute cosine similarity
        cosine_sim = cosine_sims[idx]

        # Check if suppressed (using effective tau)
        is_content = bool(content_mask[idx].item())
        exceeds_threshold = (cosine_sim >= effective_tau)
        is_suppressed = is_content and exceeds_threshold

        # Format output
        threshold_str = "YES ⚠️ " if exceeds_threshold else "NO"
        suppressed_str = "YES 🔴" if is_suppressed else "NO"

        print(f"{idx:<4} {token_str:<20} {token_type:<12} {cosine_sim:+.6f}    {threshold_str:<8} {suppressed_str:<12}")

        if is_suppressed:
            suppressed_count += 1
        if is_content:
            total_content_tokens += 1

    print("-"*100)
    print(f"Summary: {suppressed_count}/{total_content_tokens} content tokens suppressed "
          f"({100*suppressed_count/max(1, total_content_tokens):.1f}%)")

    if tau_method == "adaptive" and adaptive_tau is not None:
        ratio = adaptive_tau / tau if tau > 0 else 1.0
        print(f"Adaptive Threshold Effect: τ changed from {tau:.3f} → {adaptive_tau:.3f} "
              f"(×{ratio:.3f}, Δ = {adaptive_tau - tau:+.3f})")

    print("="*100 + "\n")


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


@torch.no_grad()
def compute_prompt_cosine_similarities(
    pipe,
    prompt: str,
    harm_vector: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarities for all tokens in prompt.
    Returns (cosine_sims, content_mask).
    """
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

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

    # Compute cosine similarities
    harm_vec_normalized = F.normalize(harm_vector.to(device), dim=-1)
    cosine_sims = torch.matmul(hidden_states, harm_vec_normalized)  # (L,)

    return cosine_sims, content_mask


# =========================
# Attention Processor
# =========================
class AdaptiveHarmSuppressionAttnProcessor(AttnProcessor2_0):
    """
    Attention processor with ADAPTIVE threshold support.
    """

    def __init__(self, harm_vector: Optional[torch.Tensor] = None,
                 tau: float = 0.15, gamma: float = 1.0, debug: bool = False):
        super().__init__()
        self.tau = tau  # Base threshold (can be overridden per prompt)
        self.gamma = gamma
        self.training = False
        self.debug = debug

        # Adaptive threshold (set per prompt)
        self.current_adaptive_tau = None

        # DEBUG: Track statistics
        self.debug_step_count = 0
        self.debug_suppression_stats = []

        # Store normalized harm vector on CPU
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
        """Set adaptive threshold for current prompt."""
        self.current_adaptive_tau = adaptive_tau

    def get_effective_tau(self) -> float:
        """Get effective threshold (adaptive if set, otherwise base)."""
        return self.current_adaptive_tau if self.current_adaptive_tau is not None else self.tau

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

        # Apply harmful concept suppression (cross-attention only)
        if is_cross_attn and self._harm_vector is not None:
            device = scores.device
            harm_vec = self._harm_vector.to(device)

            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            num_heads = scores.shape[0] // B

            # Compute cosine similarity
            context_normalized = F.normalize(encoder_hidden_states, dim=-1)
            harm_normalized = F.normalize(harm_vec, dim=-1)
            cosine_sim = torch.einsum("bkd,d->bk", context_normalized, harm_normalized)

            # Get effective threshold (adaptive or base)
            effective_tau = self.get_effective_tau()

            # Suppress tokens with similarity >= effective_tau
            suppress_mask = (cosine_sim >= effective_tau)

            # DEBUG: Track suppression statistics
            if self.debug and is_cross_attn:
                num_suppressed = suppress_mask.sum().item()
                total_tokens = suppress_mask.numel()
                avg_cosine = cosine_sim.mean().item()
                max_cosine = cosine_sim.max().item()

                self.debug_suppression_stats.append({
                    'step': self.debug_step_count,
                    'gamma': self.gamma,
                    'tau': effective_tau,
                    'num_suppressed': num_suppressed,
                    'total_tokens': total_tokens,
                    'suppression_rate': num_suppressed / max(1, total_tokens),
                    'avg_cosine': avg_cosine,
                    'max_cosine': max_cosine,
                })

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
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def print_step_debug(self, step: int):
        """Print debug info for recent step."""
        if self.debug and self.debug_suppression_stats:
            recent = self.debug_suppression_stats[-1]
            print(f"  [DEBUG Step {step:02d}] γ={recent['gamma']:.2f} | τ={recent['tau']:.3f} | "
                  f"Suppressed: {recent['num_suppressed']}/{recent['total_tokens']} ({100*recent['suppression_rate']:.1f}%) | "
                  f"Cosine: avg={recent['avg_cosine']:+.4f}, max={recent['max_cosine']:+.4f}")


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
    print(f"[INFO] Machine Unlearning - ADAPTIVE THRESHOLD MODE")
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
    # Setup Harmful Concept Suppression
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

            harm_processor = AdaptiveHarmSuppressionAttnProcessor(
                harm_vector=harm_vector,
                tau=args.base_tau,
                gamma=args.harm_gamma_start,
                debug=args.debug or args.debug_steps
            )
            pipe.unet.set_attn_processor(harm_processor)

            print(f"[INFO] Harmful concept suppression enabled")
            if args.adaptive_threshold:
                print(f"  - Threshold mode: ADAPTIVE (Multiplicative) ⭐")
                print(f"  - Base threshold (τ_base): {args.base_tau}")
                print(f"  - Central percentile: {args.central_percentile*100:.0f}%")
                print(f"  - Multiplicative factor: ×{args.tau_factor}")
                print(f"  - Formula: τ = central_mean × {args.tau_factor}")
            else:
                print(f"  - Threshold mode: FIXED")
                print(f"  - Fixed threshold (τ): {args.base_tau}")
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
            gamma = schedule_linear(step, num_steps, args.harm_gamma_start, args.harm_gamma_end)
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

        # ADAPTIVE THRESHOLD: Compute per-prompt threshold
        adaptive_tau = None
        tau_method = "fixed"

        if args.adaptive_threshold and harm_processor is not None and harm_processor.harm_vector is not None:
            # Compute cosine similarities for this prompt
            cosine_sims, content_mask = compute_prompt_cosine_similarities(
                pipe, prompt, harm_processor.harm_vector
            )

            # Extract content token similarities only
            content_sims = cosine_sims[content_mask]

            if content_sims.numel() > 0:
                # Compute adaptive threshold
                adaptive_tau = compute_adaptive_threshold(
                    content_sims,
                    central_percentile=args.central_percentile,
                    factor=args.tau_factor,
                    base_tau=args.base_tau
                )

                # Set adaptive tau in processor
                harm_processor.set_adaptive_tau(adaptive_tau)
                tau_method = "adaptive"

                # Compute ACTUAL central mean used (same logic as compute_adaptive_threshold)
                sorted_sims = torch.sort(content_sims)[0]
                n = len(sorted_sims)
                tail_fraction = (1.0 - args.central_percentile) / 2.0
                start_idx = int(n * tail_fraction)
                end_idx = int(n * (1.0 - tail_fraction))
                start_idx = max(0, start_idx)
                end_idx = min(n, end_idx)
                central_sims = sorted_sims[start_idx:end_idx]
                central_mean = float(central_sims.mean().item())

                all_mean = content_sims.mean().item()

                print(f"\n[ADAPTIVE THRESHOLD]")
                print(f"  Content tokens: {content_sims.numel()}")
                print(f"  Cosine sim range: [{content_sims.min():.3f}, {content_sims.max():.3f}]")
                print(f"  All tokens mean: {all_mean:.3f}")
                print(f"  Central mean ({int(args.central_percentile*100)}%): {central_mean:.3f} ⭐")
                print(f"  Multiplicative factor: ×{args.tau_factor}")
                print(f"  Base τ: {args.base_tau:.3f}")
                print(f"  → Adaptive τ: {adaptive_tau:.3f} (= {central_mean:.3f} × {args.tau_factor})")
            else:
                # Fallback to base threshold
                harm_processor.set_adaptive_tau(args.base_tau)
                adaptive_tau = args.base_tau
                print(f"\n[WARNING] No content tokens found, using base threshold: {args.base_tau:.3f}")
        elif harm_processor is not None:
            # Use base threshold if adaptive disabled
            harm_processor.set_adaptive_tau(args.base_tau)

        # DEBUG: Analyze prompt tokens
        if args.debug or args.debug_prompts:
            if harm_processor is not None and harm_processor.harm_vector is not None:
                debug_token_analysis(
                    pipe,
                    prompt,
                    harm_processor.harm_vector,
                    harm_concepts,
                    args.base_tau,
                    adaptive_tau=adaptive_tau,
                    tau_method=tau_method
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

        # Reset debug stats
        if harm_processor is not None:
            harm_processor.debug_suppression_stats = []
            harm_processor.debug_step_count = 0

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
            avg_cosine = np.mean([s['avg_cosine'] for s in stats])
            max_cosine_overall = max([s['max_cosine'] for s in stats])
            avg_tau = np.mean([s['tau'] for s in stats])
            print(f"  - Average suppression rate: {100*avg_rate:.1f}%")
            print(f"  - Average cosine similarity: {avg_cosine:+.4f}")
            print(f"  - Max cosine similarity: {max_cosine_overall:+.4f}")
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
