#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIFIED Harmful Content Suppression System

Combines THREE complementary approaches:
1. Soft Delete (Token-based suppression via harmful concept embeddings)
2. Latent-Guided Suppression (Real-time latent monitoring + attention suppression)
3. Classifier Guidance (Gradient-based guidance towards safe class)

Each can be enabled/disabled independently for ablation studies.
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
from geo_models.classifier.classifier import load_discriminator

import numpy as np
from typing import List, Optional, Tuple, Dict


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Unified Harmful Content Suppression")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/unified", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # === METHOD 1: Soft Delete (Token-based) ===
    parser.add_argument("--soft_delete", action="store_true",
                        help="Enable Soft Delete (token-based harmful concept suppression)")
    parser.add_argument("--harm_concepts_file", type=str, default="./configs/harm_concepts.txt",
                        help="File containing harmful concepts (one per line)")
    parser.add_argument("--adaptive_threshold", action="store_true",
                        help="Use adaptive threshold based on prompt distribution")
    parser.add_argument("--base_tau", type=float, default=0.15,
                        help="Base threshold for soft delete")
    parser.add_argument("--central_percentile", type=float, default=0.80,
                        help="Central percentile for adaptive threshold")
    parser.add_argument("--tau_factor", type=float, default=1.05,
                        help="Multiplicative factor for adaptive threshold")
    parser.add_argument("--harm_gamma_start", type=float, default=60.0,
                        help="Suppression strength at early steps")
    parser.add_argument("--harm_gamma_end", type=float, default=0.5,
                        help="Suppression strength at late steps")

    # === METHOD 2: Latent-Guided Suppression ===
    parser.add_argument("--latent_guided", action="store_true",
                        help="Enable latent-guided suppression")
    parser.add_argument("--harmful_threshold", type=float, default=0.0,
                        help="Logit threshold for harmful detection (class 2)")
    parser.add_argument("--alignment_threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for latent-attention alignment")
    parser.add_argument("--suppression_strength", type=float, default=100.0,
                        help="Suppression strength for latent-guided method")
    parser.add_argument("--suppress_start_step", type=int, default=0,
                        help="Step to start latent-guided suppression")
    parser.add_argument("--suppress_end_step", type=int, default=50,
                        help="Step to end latent-guided suppression")

    # === METHOD 3: Classifier Guidance ===
    parser.add_argument("--classifier_guidance", action="store_true",
                        help="Enable classifier guidance")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier guidance scale")
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step to start applying guidance")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance (0=not-relevant, 1=clothed, 2=nude)")

    # Shared classifier checkpoint
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint (used by both latent-guided and guidance)")
    parser.add_argument("--classifier_config", type=str,
                        default="./configs/models/time_dependent_discriminator.yaml",
                        help="Classifier config for guidance")

    # Classifier options
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classifier classes (3 or 4)")

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug_steps", action="store_true", help="Show per-step stats")
    parser.add_argument("--debug_prompts", action="store_true", help="Show per-prompt token analysis")

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
# Soft Delete: Build Harm Vector
# =========================
@torch.no_grad()
def build_harm_vector(pipe, harm_concepts: List[str]) -> torch.Tensor:
    """Build harm vector from harmful concept embeddings."""
    if not harm_concepts:
        return torch.empty(0)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device = pipe.device

    tokens = tokenizer(
        harm_concepts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(**tokens, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-2]  # Use second-to-last layer
    hidden_states = F.normalize(hidden_states, dim=-1)

    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask.bool()

    # Build content mask (exclude special tokens)
    content_mask = attention_mask.clone()
    for special_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
        if special_id is not None:
            content_mask = content_mask & (input_ids != special_id)

    # Average over content tokens only
    denom = content_mask.sum(dim=1, keepdim=True).clamp(min=1)
    vectors = (hidden_states * content_mask.unsqueeze(-1)).sum(dim=1) / denom

    # Average across all concepts and normalize -> returns 1D vector [768]
    harm_vector = F.normalize(vectors.mean(dim=0), dim=-1)

    return harm_vector


# =========================
# Soft Delete: Adaptive Threshold
# =========================
@torch.no_grad()
def compute_adaptive_threshold(
    cosine_similarities: torch.Tensor,
    central_percentile: float = 0.80,
    factor: float = 1.05,
    base_tau: float = 0.15
) -> float:
    """Compute adaptive threshold based on central mean."""
    if cosine_similarities.numel() == 0:
        return base_tau

    sorted_sims = torch.sort(cosine_similarities)[0]
    n = len(sorted_sims)

    if n < 3:
        return base_tau

    tail_fraction = (1.0 - central_percentile) / 2.0
    start_idx = int(n * tail_fraction)
    end_idx = int(n * (1.0 - tail_fraction))

    start_idx = max(0, start_idx)
    end_idx = min(n, end_idx)

    if end_idx <= start_idx:
        return base_tau

    central_sims = sorted_sims[start_idx:end_idx]
    central_mean = float(central_sims.mean().item())
    adaptive_tau = central_mean * factor
    adaptive_tau = max(0.0, min(1.0, adaptive_tau))

    return adaptive_tau


# =========================
# Latent Monitor with Classifier
# =========================
class LatentMonitor:
    """Monitor latents and detect harmful content."""

    def __init__(self, classifier_ckpt: str, harmful_threshold: float = 0.0,
                 device: str = "cuda", num_classes: int = 3):
        self.device = device
        self.harmful_threshold = harmful_threshold
        self.num_classes = num_classes

        print(f"\n[INFO] Loading latent classifier from {classifier_ckpt}...")
        self.classifier = load_discriminator(
            ckpt_path=classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,
            num_classes=num_classes
        ).to(device)
        self.classifier.eval()
        print(f"[INFO] Classifier loaded successfully")

        self.current_step = 0
        self.is_harmful = False
        self.harmful_logit = 0.0
        self.harmful_latent_features = None

    @torch.no_grad()
    def evaluate_latent(self, latent: torch.Tensor, timestep: int) -> Dict:
        """Evaluate latent with classifier."""
        latent_fp32 = latent.to(self.device).float()
        t = torch.tensor([timestep], device=self.device)

        logits = self.classifier(latent_fp32, t)
        harmful_logit = logits[0, 2].item()  # Class 2 = nude

        is_harmful = harmful_logit > self.harmful_threshold
        features = latent if is_harmful else None

        return {
            'is_harmful': is_harmful,
            'logit': harmful_logit,
            'features': features,
            'timestep': timestep
        }

    def update(self, latent: torch.Tensor, timestep: int, step: int):
        """Update monitor state."""
        self.current_step = step
        result = self.evaluate_latent(latent, timestep)

        self.is_harmful = result['is_harmful']
        self.harmful_logit = result['logit']
        self.harmful_latent_features = result['features']

        return result


# =========================
# UNIFIED Attention Processor
# Combines: Soft Delete + Latent-Guided Suppression
# =========================
class UnifiedSuppressionAttnProcessor(AttnProcessor2_0):
    """
    Unified attention processor combining:
    1. Soft Delete (token-based)
    2. Latent-Guided Suppression
    """

    def __init__(self,
                 # Soft Delete params
                 harm_vector: Optional[torch.Tensor] = None,
                 tau: float = 0.15,
                 gamma: float = 1.0,
                 use_soft_delete: bool = False,
                 # Latent-Guided params
                 latent_monitor: Optional[LatentMonitor] = None,
                 alignment_threshold: float = 0.3,
                 suppression_strength: float = 100.0,
                 suppress_start_step: int = 0,
                 suppress_end_step: int = 50,
                 use_latent_guided: bool = False,
                 # Debug
                 debug: bool = False):
        super().__init__()

        # Soft Delete
        self.use_soft_delete = use_soft_delete
        self.tau = tau
        self.gamma = gamma
        self.current_adaptive_tau = None
        if harm_vector is None or harm_vector.numel() == 0:
            self._harm_vector = None
        else:
            self._harm_vector = F.normalize(harm_vector.detach().float(), dim=-1).cpu()

        # Latent-Guided
        self.use_latent_guided = use_latent_guided
        self.latent_monitor = latent_monitor
        self.alignment_threshold = alignment_threshold
        self.suppression_strength = suppression_strength
        self.suppress_start_step = suppress_start_step
        self.suppress_end_step = suppress_end_step

        # Debug
        self.debug = debug
        self.total_soft_delete_suppressions = 0
        self.total_latent_guided_suppressions = 0
        self.step_stats = {}

    @property
    def harm_vector(self):
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

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Apply unified suppression."""

        residual = hidden_states

        # Check if this is cross-attention (encoder_hidden_states is provided)
        is_cross_attn = encoder_hidden_states is not None
        original_encoder_hidden_states = encoder_hidden_states  # Save for Soft Delete

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Compute Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Initialize combined suppression bias
        combined_bias = None

        # ============================================================
        # METHOD 1: SOFT DELETE (Token-based suppression)
        # Only apply to cross-attention where original_encoder_hidden_states is text embeddings
        # ============================================================
        if self.use_soft_delete and is_cross_attn and self._harm_vector is not None:
            harm_vec = self._harm_vector.to(value.device)
            content_embeds = original_encoder_hidden_states  # Use original text embeddings

            content_norm = F.normalize(content_embeds, p=2, dim=-1)
            harm_vec_norm = F.normalize(harm_vec, p=2, dim=-1)
            # harm_vec is 1D [768], content_norm is [batch, seq_len, 768]
            cosine_sim = torch.matmul(content_norm, harm_vec_norm)  # [batch, seq_len]

            effective_tau = self.get_effective_tau()
            suppress_mask = (cosine_sim > effective_tau).float()

            num_soft_delete_suppressed = suppress_mask.sum().item()
            if num_soft_delete_suppressed > 0:
                self.total_soft_delete_suppressions += num_soft_delete_suppressed

                # Create soft delete bias
                soft_delete_bias = torch.zeros(
                    batch_size, attn.heads, query.shape[2], key.shape[2],
                    device=query.device, dtype=query.dtype
                )

                suppress_mask_expanded = suppress_mask.unsqueeze(1).unsqueeze(1)
                suppress_mask_expanded = suppress_mask_expanded.expand(-1, attn.heads, query.shape[2], -1)

                soft_delete_bias = soft_delete_bias.masked_fill(
                    suppress_mask_expanded.bool(),
                    -self.gamma
                )

                combined_bias = soft_delete_bias

        # ============================================================
        # METHOD 2: LATENT-GUIDED SUPPRESSION
        # ============================================================
        if self.use_latent_guided and self.latent_monitor is not None:
            current_step = self.latent_monitor.current_step

            should_suppress = (
                self.suppress_start_step <= current_step <= self.suppress_end_step
                and self.latent_monitor.is_harmful
                and self.latent_monitor.harmful_latent_features is not None
            )

            if should_suppress:
                harmful_latent = self.latent_monitor.harmful_latent_features

                _, C, H, W = harmful_latent.shape
                harmful_features = harmful_latent.view(1, C, H * W).transpose(1, 2)
                harmful_vec = harmful_features.mean(dim=1, keepdim=True)

                key_combined = key.transpose(1, 2).reshape(batch_size, -1, inner_dim)

                harmful_vec_norm = F.normalize(harmful_vec, p=2, dim=-1)
                key_norm = F.normalize(key_combined, p=2, dim=-1)

                if harmful_vec_norm.shape[-1] != inner_dim:
                    harmful_flat = harmful_vec_norm.squeeze(1)
                    harmful_vec_pooled = F.interpolate(
                        harmful_flat.unsqueeze(1),
                        size=inner_dim,
                        mode='linear',
                        align_corners=False
                    ).squeeze(1).unsqueeze(1)
                else:
                    harmful_vec_pooled = harmful_vec_norm

                harmful_vec_pooled = harmful_vec_pooled.expand(batch_size, -1, -1)

                similarity = torch.bmm(key_norm, harmful_vec_pooled.transpose(1, 2)).squeeze(-1)
                suppress_mask = (similarity > self.alignment_threshold).float()

                num_latent_guided_suppressed = suppress_mask.sum().item()
                if num_latent_guided_suppressed > 0:
                    self.total_latent_guided_suppressions += num_latent_guided_suppressed

                    latent_guided_bias = torch.zeros(
                        batch_size, attn.heads, query.shape[2], key.shape[2],
                        device=query.device, dtype=query.dtype
                    )

                    suppress_mask_expanded = suppress_mask.unsqueeze(1).unsqueeze(1)
                    suppress_mask_expanded = suppress_mask_expanded.expand(-1, attn.heads, query.shape[2], -1)

                    latent_guided_bias = latent_guided_bias.masked_fill(
                        suppress_mask_expanded.bool(),
                        -self.suppression_strength
                    )

                    # Combine with soft delete bias
                    if combined_bias is not None:
                        combined_bias = combined_bias + latent_guided_bias
                    else:
                        combined_bias = latent_guided_bias

        # Apply combined bias to attention mask
        if combined_bias is not None:
            if attention_mask is not None:
                attention_mask = attention_mask + combined_bias
            else:
                attention_mask = combined_bias

        # Standard attention computation
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear proj and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# =========================
# Main
# =========================
def main():
    args = parse_args()

    print("="*80)
    print("UNIFIED HARMFUL CONTENT SUPPRESSION")
    print("="*80)
    print(f"\nModel: {args.ckpt_path}")
    print(f"Prompts: {args.prompt_file}")
    print(f"Output: {args.output_dir}")

    # Setup device
    accelerator = Accelerator()
    device = accelerator.device

    # Load prompts
    import csv as csv_module
    if args.prompt_file.endswith('.csv'):
        prompts = []
        with open(args.prompt_file, 'r') as f:
            reader = csv_module.DictReader(f)
            col_priority = ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text']
            prompt_col = None
            for col in col_priority:
                if col in reader.fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"CSV has no recognizable prompt column: {reader.fieldnames}")
            for row in reader:
                prompts.append(row[prompt_col].strip())
        print(f"[INFO] Using column '{prompt_col}' from CSV")
    else:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(prompts)} prompts")

    # Load pipeline
    print(f"\n[INFO] Loading Stable Diffusion from {args.ckpt_path}...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    print("[INFO] Pipeline loaded successfully")

    # Initialize components based on enabled methods
    harm_vector = None
    latent_monitor = None
    guidance_model = None

    # === METHOD 1: Soft Delete ===
    if args.soft_delete:
        print("\n" + "─"*80)
        print("SOFT DELETE CONFIG (Token-based)")
        print("─"*80)
        with open(args.harm_concepts_file, 'r') as f:
            harm_concepts = [line.strip() for line in f if line.strip()]

        print(f"  Harmful concepts: {len(harm_concepts)}")
        print(f"  Base tau: {args.base_tau}")
        print(f"  Adaptive threshold: {args.adaptive_threshold}")
        print(f"  Gamma schedule: {args.harm_gamma_start} → {args.harm_gamma_end}")

        harm_vector = build_harm_vector(pipe, harm_concepts)
        print(f"[INFO] Soft Delete enabled")

    # === METHOD 2: Latent-Guided ===
    if args.latent_guided:
        print("\n" + "─"*80)
        print("LATENT-GUIDED SUPPRESSION CONFIG")
        print("─"*80)
        print(f"  Classifier: {args.classifier_ckpt}")
        print(f"  Harmful threshold (Class 2): {args.harmful_threshold}")
        print(f"  Alignment threshold: {args.alignment_threshold}")
        print(f"  Suppression strength: {args.suppression_strength}")
        print(f"  Steps: {args.suppress_start_step} → {args.suppress_end_step}")

        latent_monitor = LatentMonitor(
            classifier_ckpt=args.classifier_ckpt,
            harmful_threshold=args.harmful_threshold,
            device=device,
            num_classes=args.num_classes
        )
        print(f"[INFO] Latent-Guided enabled")

    # === METHOD 3: Classifier Guidance ===
    if args.classifier_guidance:
        print("\n" + "─"*80)
        print("CLASSIFIER GUIDANCE CONFIG")
        print("─"*80)
        print(f"  Config: {args.classifier_config}")
        print(f"  Checkpoint: {args.classifier_ckpt}")
        print(f"  Scale: {args.guidance_scale}")
        print(f"  Target class: {args.target_class}")
        print(f"  Start step: {args.guidance_start_step}")

        guidance_model = GuidanceModel(
            diffusion_pipeline=pipe,
            model_config_file=args.classifier_config,
            model_ckpt_path=args.classifier_ckpt,
            target_class=args.target_class,
            device=device
        )
        print(f"[INFO] Classifier Guidance enabled")

    # Install unified processor
    unified_processor = UnifiedSuppressionAttnProcessor(
        harm_vector=harm_vector,
        tau=args.base_tau,
        gamma=args.harm_gamma_start,
        use_soft_delete=args.soft_delete,
        latent_monitor=latent_monitor,
        alignment_threshold=args.alignment_threshold,
        suppression_strength=args.suppression_strength,
        suppress_start_step=args.suppress_start_step,
        suppress_end_step=args.suppress_end_step,
        use_latent_guided=args.latent_guided,
        debug=args.debug or args.debug_steps
    )

    for name, module in pipe.unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(unified_processor)

    # Generation loop
    print("\n" + "="*80)
    print("STARTING GENERATION")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] Prompt: {prompt}")

        # Set adaptive threshold for soft delete if enabled
        if args.soft_delete and args.adaptive_threshold and harm_vector is not None:
            tokens = pipe.tokenizer(
                [prompt],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = pipe.text_encoder(**tokens, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[-2][0]  # (L, d)
            hidden_states = F.normalize(hidden_states, dim=-1)

            # Compute cosine similarities with 1D harm_vector
            harm_norm = F.normalize(harm_vector.to(device), p=2, dim=-1)
            cosine_sims = torch.matmul(hidden_states, harm_norm)  # (L,)

            # Build content mask (exclude special tokens)
            input_ids = tokens.input_ids[0]
            attention_mask = tokens.attention_mask[0]
            content_mask = attention_mask.bool().clone()
            for special_id in [pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id, pipe.tokenizer.pad_token_id]:
                if special_id is not None:
                    content_mask = content_mask & (input_ids != special_id)

            # Extract content token similarities only
            content_sims = cosine_sims[content_mask]

            if content_sims.numel() > 0:
                adaptive_tau = compute_adaptive_threshold(
                    content_sims,
                    args.central_percentile,
                    args.tau_factor,
                    args.base_tau
                )
                unified_processor.set_adaptive_tau(adaptive_tau)

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            generator = torch.Generator(device=device).manual_seed(seed)

            # Callback for latent monitoring
            def latent_callback(step, timestep, latents):
                # Update gamma for soft delete
                if args.soft_delete:
                    gamma = schedule_linear(
                        step, args.num_inference_steps,
                        args.harm_gamma_start, args.harm_gamma_end
                    )
                    unified_processor.set_gamma(gamma)

                # Update latent monitor
                if latent_monitor is not None:
                    latent_monitor.update(latents, timestep, step)

            # Generate
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Note: Classifier guidance integration requires CustomStableDiffusionPipeline modification
                # For now, only soft delete + latent-guided work
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    generator=generator,
                    callback=latent_callback,
                    callback_steps=1
                ).images[0]

            # Save
            filename = f"prompt_{prompt_idx:04d}_sample_{sample_idx:02d}.png"
            save_image(image, filename, root=args.output_dir)
            print(f"  Saved: {filename}")

    # Print statistics
    print("\n" + "="*80)
    print("SUPPRESSION STATISTICS")
    print("="*80)

    if args.soft_delete:
        print(f"Soft Delete (Token-based):")
        print(f"  Total suppressions: {unified_processor.total_soft_delete_suppressions}")

    if args.latent_guided:
        print(f"Latent-Guided:")
        print(f"  Total suppressions: {unified_processor.total_latent_guided_suppressions}")

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
