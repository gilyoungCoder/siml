#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latent-Guided Harmful Content Suppression

Key Innovation:
  - Token-based suppression (기존) → Latent-based suppression (새로운 방식)
  - Real-time latent monitoring with classifier
  - Direct attention suppression for latent-aligned tokens
  - Hard blocking when harmful content is detected

Mechanism:
  1. Monitor latent at each diffusion step with classifier
  2. If harmful detected (class 1 logit > threshold):
     - Extract latent features
     - Compute cosine similarity with attention values (K projection)
     - Hard suppress attentions with high similarity to harmful latent
  3. Continue generation with suppressed harmful patterns
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
    parser = ArgumentParser(description="Latent-Guided Harmful Content Suppression")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained model checkpoint")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/latent_guided", help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Latent-based suppression
    parser.add_argument("--latent_suppress", action="store_true",
                        help="Enable latent-based harmful suppression")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class_clothed/checkpoint/step_10700/classifier.pth",
                        help="Classifier checkpoint for latent evaluation")
    parser.add_argument("--harmful_threshold", type=float, default=0.0,
                        help="Logit threshold for harmful detection (class 1)")
    parser.add_argument("--suppression_strength", type=float, default=100.0,
                        help="Suppression strength (higher = harder block)")
    parser.add_argument("--alignment_threshold", type=float, default=0.3,
                        help="Cosine similarity threshold for latent-attention alignment")

    # Suppression schedule
    parser.add_argument("--suppress_start_step", type=int, default=0,
                        help="Step to start suppression")
    parser.add_argument("--suppress_end_step", type=int, default=50,
                        help="Step to end suppression")

    # Classifier Guidance (optional, can be used together)
    parser.add_argument("--classifier_guidance", action="store_true", help="Enable classifier guidance")
    parser.add_argument("--classifier_config", type=str,
                        default="./configs/models/time_dependent_discriminator.yaml",
                        help="Classifier model configuration file")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--guidance_start_step", type=int, default=1,
                        help="Step to start applying guidance")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class for guidance")

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug_steps", action="store_true", help="Show per-step suppression stats")

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


# =========================
# Latent Monitor with Classifier
# =========================
class LatentMonitor:
    """
    Monitor latents at each step and detect harmful content using classifier.
    Stores harmful latent features for attention suppression.
    """

    def __init__(self, classifier_ckpt: str, harmful_threshold: float = 0.0,
                 device: str = "cuda", num_classes: int = 3):
        self.device = device
        self.harmful_threshold = harmful_threshold
        self.num_classes = num_classes

        # Load classifier
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

        # Storage for harmful detection
        self.current_step = 0
        self.is_harmful = False
        self.harmful_logit = 0.0
        self.harmful_latent_features = None

    @torch.no_grad()
    def evaluate_latent(self, latent: torch.Tensor, timestep: int) -> Dict:
        """
        Evaluate latent with classifier and extract features if harmful.

        Returns:
            dict with 'is_harmful', 'logit', 'features'
        """
        # Convert to float32 for classifier
        latent_fp32 = latent.to(self.device).float()

        # Create timestep tensor
        t = torch.tensor([timestep], device=self.device)

        # Get classifier prediction
        logits = self.classifier(latent_fp32, t)  # [B, num_classes]
        harmful_logit = logits[0, 2].item()  # Class 2 = harmful (nude people)

        # Check if harmful
        is_harmful = harmful_logit > self.harmful_threshold

        # Extract features (use latent itself as feature representation)
        # In a more sophisticated version, could extract intermediate features from classifier
        features = latent if is_harmful else None

        result = {
            'is_harmful': is_harmful,
            'logit': harmful_logit,
            'features': features,
            'timestep': timestep
        }

        return result

    def update(self, latent: torch.Tensor, timestep: int, step: int):
        """Update monitor state with new latent."""
        self.current_step = step

        result = self.evaluate_latent(latent, timestep)

        self.is_harmful = result['is_harmful']
        self.harmful_logit = result['logit']
        self.harmful_latent_features = result['features']

        return result


# =========================
# Latent-Guided Attention Processor
# =========================
class LatentGuidedSuppressionAttnProcessor(AttnProcessor2_0):
    """
    Attention processor that suppresses patterns aligned with harmful latent features.

    When harmful latent is detected:
    1. Compute attention keys (K = Wk @ latent_features)
    2. Compute cosine similarity between K and harmful latent features
    3. Hard suppress attentions with high similarity
    """

    def __init__(self, latent_monitor: LatentMonitor,
                 alignment_threshold: float = 0.3,
                 suppression_strength: float = 100.0,
                 suppress_start_step: int = 0,
                 suppress_end_step: int = 50,
                 debug: bool = False):
        super().__init__()

        self.latent_monitor = latent_monitor
        self.alignment_threshold = alignment_threshold
        self.suppression_strength = suppression_strength
        self.suppress_start_step = suppress_start_step
        self.suppress_end_step = suppress_end_step
        self.debug = debug

        # Statistics
        self.total_suppressions = 0
        self.step_suppressions = {}

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
        """
        Apply latent-guided suppression to attention computation.
        """
        residual = hidden_states

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

        # ====================================================
        # LATENT-GUIDED SUPPRESSION
        # ====================================================
        current_step = self.latent_monitor.current_step

        # Check if we should apply suppression
        should_suppress = (
            self.suppress_start_step <= current_step <= self.suppress_end_step
            and self.latent_monitor.is_harmful
            and self.latent_monitor.harmful_latent_features is not None
        )

        if should_suppress:
            # Get harmful latent features [1, C, H, W] (from monitor)
            harmful_latent = self.latent_monitor.harmful_latent_features

            # Flatten latent to [1, C, H*W] then [1, H*W, C]
            _, C, H, W = harmful_latent.shape
            harmful_features = harmful_latent.view(1, C, H * W).transpose(1, 2)  # [1, H*W, C]

            # Compute average feature vector across spatial locations
            harmful_vec = harmful_features.mean(dim=1, keepdim=True)  # [1, 1, C]

            # Project keys back to original dimension for comparison
            # key: [B, num_heads, seq_len, head_dim]
            # batch_size might be 2 (CFG: conditional + unconditional)
            key_combined = key.transpose(1, 2).reshape(batch_size, -1, inner_dim)  # [B, seq_len, inner_dim]

            # Normalize for cosine similarity
            harmful_vec_norm = F.normalize(harmful_vec, p=2, dim=-1)  # [1, 1, C]
            key_norm = F.normalize(key_combined, p=2, dim=-1)  # [B, seq_len, inner_dim]

            # Match dimensions between harmful_vec and key
            # harmful_vec_norm: [1, 1, C] where C=4 (latent channels)
            # key_norm: [B, seq_len, inner_dim] where inner_dim=320 (attention dim)
            # We need to project C -> inner_dim

            if harmful_vec_norm.shape[-1] != inner_dim:
                # Use linear interpolation to match dimensions
                # Reshape to [1, C] for interpolation
                harmful_flat = harmful_vec_norm.squeeze(1)  # [1, C]

                # Interpolate from C dimensions to inner_dim dimensions
                harmful_vec_pooled = F.interpolate(
                    harmful_flat.unsqueeze(1),  # [1, 1, C]
                    size=inner_dim,
                    mode='linear',
                    align_corners=False
                ).squeeze(1).unsqueeze(1)  # [1, 1, inner_dim]
            else:
                harmful_vec_pooled = harmful_vec_norm  # [1, 1, inner_dim]

            # Expand harmful_vec to match batch_size (for CFG, batch_size=2)
            harmful_vec_pooled = harmful_vec_pooled.expand(batch_size, -1, -1)  # [B, 1, inner_dim]

            # Compute similarity: [B, seq_len]
            similarity = torch.bmm(key_norm, harmful_vec_pooled.transpose(1, 2)).squeeze(-1)

            # Create suppression mask
            suppress_mask = (similarity > self.alignment_threshold).float()  # [B, seq_len]

            # Count suppressions
            num_suppressed = suppress_mask.sum().item()
            if num_suppressed > 0:
                self.total_suppressions += num_suppressed
                if current_step not in self.step_suppressions:
                    self.step_suppressions[current_step] = 0
                self.step_suppressions[current_step] += num_suppressed

                if self.debug:
                    print(f"  [Step {current_step}] Harmful detected (logit={self.latent_monitor.harmful_logit:.3f}), "
                          f"suppressing {num_suppressed}/{sequence_length} tokens (threshold={self.alignment_threshold})")

            # Apply suppression to attention scores
            # We'll create a bias mask to add to attention scores
            # Shape needs to be [B, num_heads, query_len, key_len]
            suppress_bias = torch.zeros(
                batch_size, attn.heads, query.shape[2], key.shape[2],
                device=query.device, dtype=query.dtype
            )

            # Expand suppress_mask to all heads and queries
            suppress_mask_expanded = suppress_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, seq_len]
            suppress_mask_expanded = suppress_mask_expanded.expand(-1, attn.heads, query.shape[2], -1)

            # Apply strong negative bias to suppress harmful attentions
            suppress_bias = suppress_bias.masked_fill(
                suppress_mask_expanded.bool(),
                -self.suppression_strength
            )

            # Add to attention_mask (if exists) or create new
            if attention_mask is not None:
                attention_mask = attention_mask + suppress_bias
            else:
                attention_mask = suppress_bias

        # Continue with standard attention computation
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# =========================
# Main Generation Function
# =========================
def main():
    args = parse_args()

    print("="*80)
    print("LATENT-GUIDED HARMFUL CONTENT SUPPRESSION")
    print("="*80)
    print(f"\nModel: {args.ckpt_path}")
    print(f"Prompts: {args.prompt_file}")
    print(f"Output: {args.output_dir}")

    # Setup device
    accelerator = Accelerator()
    device = accelerator.device

    # Load prompts
    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"\nLoaded {len(prompts)} prompts")

    # Load Stable Diffusion pipeline
    print(f"\n[INFO] Loading Stable Diffusion from {args.ckpt_path}...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    print("[INFO] Pipeline loaded successfully")

    # Setup latent monitoring and suppression
    latent_monitor = None
    if args.latent_suppress:
        print("\n" + "─"*80)
        print("LATENT-BASED SUPPRESSION CONFIG")
        print("─"*80)
        print(f"  Classifier: {args.classifier_ckpt}")
        print(f"  Harmful threshold: {args.harmful_threshold}")
        print(f"  Alignment threshold: {args.alignment_threshold}")
        print(f"  Suppression strength: {args.suppression_strength}")
        print(f"  Suppression steps: {args.suppress_start_step} → {args.suppress_end_step}")

        latent_monitor = LatentMonitor(
            classifier_ckpt=args.classifier_ckpt,
            harmful_threshold=args.harmful_threshold,
            device=device,
            num_classes=3
        )

        # Install latent-guided processor
        latent_processor = LatentGuidedSuppressionAttnProcessor(
            latent_monitor=latent_monitor,
            alignment_threshold=args.alignment_threshold,
            suppression_strength=args.suppression_strength,
            suppress_start_step=args.suppress_start_step,
            suppress_end_step=args.suppress_end_step,
            debug=args.debug or args.debug_steps
        )

        # Set processor for all attention layers
        for name, module in pipe.unet.named_modules():
            if isinstance(module, Attention):
                module.set_processor(latent_processor)

        print(f"[INFO] Latent-guided suppression enabled")

    # Setup classifier guidance (optional)
    guidance_model = None
    if args.classifier_guidance:
        print("\n" + "─"*80)
        print("CLASSIFIER GUIDANCE CONFIG")
        print("─"*80)
        print(f"  Scale: {args.guidance_scale}")
        print(f"  Target class: {args.target_class}")
        print(f"  Start step: {args.guidance_start_step}")
        print(f"  Config: {args.classifier_config}")
        print(f"  Checkpoint: {args.classifier_ckpt}")

        # GuidanceModel uses the correct signature
        guidance_model = GuidanceModel(
            diffusion_pipeline=pipe,
            model_config_file=args.classifier_config,
            model_ckpt_path=args.classifier_ckpt,
            target_class=args.target_class,
            device=device
        )
        print(f"[INFO] Classifier guidance enabled")

    # Generation loop
    print("\n" + "="*80)
    print("STARTING GENERATION")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx+1}/{len(prompts)}] Prompt: {prompt}")

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            generator = torch.Generator(device=device).manual_seed(seed)

            # Define callback for latent monitoring
            def latent_callback(step, timestep, latents):
                if latent_monitor is not None:
                    latent_monitor.update(latents, timestep, step)

            # Generate
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if guidance_model is not None:
                    # With classifier guidance
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.cfg_scale,
                        generator=generator,
                        callback=latent_callback,
                        callback_steps=1,
                        guidance_model=guidance_model,
                        guidance_start_step=args.guidance_start_step
                    ).images[0]
                else:
                    # Without classifier guidance
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
    if latent_monitor is not None and hasattr(latent_processor, 'total_suppressions'):
        print("\n" + "="*80)
        print("SUPPRESSION STATISTICS")
        print("="*80)
        print(f"Total suppressed tokens: {latent_processor.total_suppressions}")
        print(f"Steps with suppression: {len(latent_processor.step_suppressions)}")
        if latent_processor.step_suppressions:
            print("\nPer-step suppressions:")
            for step, count in sorted(latent_processor.step_suppressions.items()):
                print(f"  Step {step:2d}: {count} tokens")

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
