#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Unlearning with DEBUG LOGGING - Clean Implementation

Features:
  1. Attention Manipulation: Suppressing harmful concepts via cosine similarity
  2. Classifier Guidance: Guiding generation towards "clothed people" class
  3. DEBUG: Per-token analysis with cosine similarities and threshold detection

No parameter updates to the diffusion model.
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
from typing import List, Optional


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Machine Unlearning with DEBUG: Attention Manipulation + Classifier Guidance")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="output_img/unlearning_debug", help="Output directory for generated images")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Harmful concept suppression (Attention Manipulation)
    parser.add_argument("--harm_suppress", action="store_true", help="Enable harmful concept suppression")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt",
                        help="File containing harmful concepts to suppress (one per line)")
    parser.add_argument("--harm_tau", type=float, default=0.15,
                        help="Cosine similarity threshold for harmful concept detection")
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
# DEBUG: Token Analysis
# =========================
@torch.no_grad()
def debug_token_analysis(pipe, prompt: str, harm_vector: torch.Tensor,
                         harm_concepts: List[str], tau: float):
    """
    DEBUG: Analyze each token in the prompt for harmful concept similarity.
    Shows cosine similarity with harm vector and threshold crossing.
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
    hidden_states = outputs.hidden_states[-2][0]  # (L, d), second-to-last layer
    hidden_states = F.normalize(hidden_states, dim=-1)

    # Get token strings
    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    # Normalize harm vector
    harm_vec_normalized = F.normalize(harm_vector.to(device), dim=-1)

    # Build content mask (exclude special tokens)
    content_mask = attention_mask.bool().clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    # Print header
    print(f"\nHarm Concepts: {', '.join(harm_concepts)}")
    print(f"Threshold (τ): {tau:.3f}")
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
        is_special = False
        token_type = "CONTENT"
        if tokenizer.bos_token_id and token_id == tokenizer.bos_token_id:
            token_type = "BOS/SOT"
            is_special = True
        elif tokenizer.eos_token_id and token_id == tokenizer.eos_token_id:
            token_type = "EOS/EOT"
            is_special = True
        elif tokenizer.pad_token_id and token_id == tokenizer.pad_token_id:
            token_type = "PAD"
            is_special = True

        # Compute cosine similarity
        cosine_sim = float(torch.dot(hidden_states[idx], harm_vec_normalized).item())

        # Check if suppressed
        is_content = bool(content_mask[idx].item())
        exceeds_threshold = (cosine_sim >= tau)
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
    print(f"Summary: {suppressed_count}/{total_content_tokens} content tokens suppressed ({100*suppressed_count/max(1, total_content_tokens):.1f}%)")
    print("="*100 + "\n")


# =========================
# Vector Building
# =========================
@torch.no_grad()
def build_harm_vector(pipe, concepts: List[str]) -> Optional[torch.Tensor]:
    """
    Build a normalized vector representing harmful concepts.
    Uses mean pooling of text encoder embeddings (excluding special tokens).
    """
    if not concepts:
        return None

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    # Tokenize harmful concepts
    tokens = tokenizer(
        concepts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Get text encoder embeddings
    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-2]  # Use second-to-last layer
    hidden_states = F.normalize(hidden_states, dim=-1)

    # Build content mask (exclude BOS/EOS/PAD tokens)
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask.bool()

    content_mask = attention_mask.clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    # Mean pooling over valid tokens
    denom = content_mask.sum(dim=1, keepdim=True).clamp(min=1)
    vectors = (hidden_states * content_mask.unsqueeze(-1)).sum(dim=1) / denom  # (N, d)

    # Average across all concepts
    harm_vector = F.normalize(vectors.mean(dim=0), dim=-1)  # (d,)

    return harm_vector


# =========================
# Attention Processor
# =========================
class HarmSuppressionAttnProcessor(AttnProcessor2_0):
    """
    Custom attention processor that suppresses harmful concepts in cross-attention.
    Reduces attention scores for tokens with high cosine similarity to harmful concept vector.
    """

    def __init__(self, harm_vector: Optional[torch.Tensor] = None,
                 tau: float = 0.15, gamma: float = 1.0, debug: bool = False):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.training = False
        self.debug = debug

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

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:

        # Standard attention computation
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
        query = attn.head_to_batch_dim(query)  # (B*H, Q, d_h)
        key = attn.head_to_batch_dim(key)      # (B*H, K, d_h)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) * attn.scale  # (B*H, Q, K)

        # Apply harmful concept suppression (only for cross-attention)
        if is_cross_attn and self._harm_vector is not None:
            device = scores.device
            harm_vec = self._harm_vector.to(device)

            B = encoder_hidden_states.shape[0]
            Q = scores.shape[1]
            K = scores.shape[2]
            num_heads = scores.shape[0] // B

            # Compute cosine similarity between context embeddings and harm vector
            context_normalized = F.normalize(encoder_hidden_states, dim=-1)  # (B, K, d)
            harm_normalized = F.normalize(harm_vec, dim=-1)  # (d,)

            cosine_sim = torch.einsum("bkd,d->bk", context_normalized, harm_normalized)  # (B, K)

            # Suppress tokens with similarity >= tau
            suppress_mask = (cosine_sim >= self.tau)  # (B, K)

            # DEBUG: Track suppression statistics
            if self.debug and is_cross_attn:
                num_suppressed = suppress_mask.sum().item()
                total_tokens = suppress_mask.numel()
                avg_cosine = cosine_sim.mean().item()
                max_cosine = cosine_sim.max().item()

                self.debug_suppression_stats.append({
                    'step': self.debug_step_count,
                    'gamma': self.gamma,
                    'num_suppressed': num_suppressed,
                    'total_tokens': total_tokens,
                    'suppression_rate': num_suppressed / max(1, total_tokens),
                    'avg_cosine': avg_cosine,
                    'max_cosine': max_cosine,
                })

            if suppress_mask.any():
                # Calculate suppression weight
                weight = cosine_sim.clamp(min=0.0) * suppress_mask.float()  # (B, K)

                # Expand to match score dimensions
                weight_expanded = weight[:, None, :].expand(B, Q, K).repeat_interleave(num_heads, dim=0)  # (B*H, Q, K)

                # Suppress attention scores
                scores = scores - (weight_expanded * self.gamma)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            scores = scores + attention_mask

        # Softmax to get attention probabilities
        attn_probs = F.softmax(scores, dim=-1)

        # Apply dropout
        if isinstance(attn.dropout, nn.Dropout):
            attn_probs = attn.dropout(attn_probs)
        else:
            p = float(attn.dropout) if isinstance(attn.dropout, (int, float)) else 0.0
            attn_probs = F.dropout(attn_probs, p=p, training=False)

        # Compute output
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def print_step_debug(self, step: int):
        """Print debug info for recent step."""
        if self.debug and self.debug_suppression_stats:
            recent = self.debug_suppression_stats[-1]
            print(f"  [DEBUG Step {step:02d}] γ={recent['gamma']:.2f} | "
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
    print(f"[INFO] Machine Unlearning - DEBUG MODE")
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

            harm_processor = HarmSuppressionAttnProcessor(
                harm_vector=harm_vector,
                tau=args.harm_tau,
                gamma=args.harm_gamma_start,
                debug=args.debug or args.debug_steps
            )
            pipe.unet.set_attn_processor(harm_processor)
            print(f"[INFO] Harmful concept suppression enabled")
            print(f"  - Threshold (τ): {args.harm_tau}")
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
            1,  # num_classes (not used for inference)
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

        # DEBUG: Analyze prompt tokens
        if args.debug or args.debug_prompts:
            if harm_processor is not None and harm_processor.harm_vector is not None:
                debug_token_analysis(
                    pipe,
                    prompt,
                    harm_processor.harm_vector,
                    harm_concepts,
                    args.harm_tau
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
            print(f"  - Average suppression rate: {100*avg_rate:.1f}%")
            print(f"  - Average cosine similarity: {avg_cosine:+.4f}")
            print(f"  - Max cosine similarity: {max_cosine_overall:+.4f}")

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
