"""
Self-Healing Guided Diffusion (SHGD) Pipeline v2

Training-free safe generation via:
1. Dual-Anchor CFG: directional shift from harmful -> safe concepts
2. Guided-then-Heal: strong guidance + diffusion prior healing
3. Self-Consistency Monitor: adaptive off-manifold detection

v2 additions:
4. Negative Concept Amplification: separate strong negative term for harmful concepts
5. Sample-Adaptive Guidance: per-sample harm detection via x0 prediction
6. Multi-Round Guide-Heal: iterative refinement for stubborn cases
7. Progressive Guidance Schedule: cosine/linear decay of guidance strength
8. Cross-Attention Suppression: direct attention manipulation on harmful tokens
"""

import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class SHGDPipeline(StableDiffusionPipeline):
    """
    Stable Diffusion pipeline with Self-Healing Guided Diffusion (SHGD).

    Extends the standard SD pipeline with:
    - Dual-Anchor CFG for directional concept shifting
    - Guided-then-Heal scheduling with re-noising
    - Self-consistency monitoring for adaptive healing
    - Negative concept amplification
    - Sample-adaptive guidance
    - Multi-round guide-heal
    - Cross-attention suppression
    """

    def _register_attn_hooks(self, harmful_token_ids, suppress_scale):
        """Register hooks on cross-attention layers to suppress harmful tokens."""
        self._attn_hooks = []
        self._attn_suppress_token_ids = harmful_token_ids
        self._attn_suppress_scale = suppress_scale
        self._attn_manipulation_active = False

        def make_hook(module):
            def hook_fn(module, args, output):
                if not self._attn_manipulation_active:
                    return output
                # output is the attention output; we need to intercept during forward
                # This hook modifies the hidden states after attention
                return output
            return hook_fn

        # We'll use a different approach: modify the text embeddings directly
        # by zeroing out harmful token embeddings during guided steps
        # This is more reliable than hooking into attention layers

    def _get_harmful_token_mask(self, prompt, harmful_words, device):
        """
        Create a mask identifying tokens in the prompt that relate to harmful content.
        Returns a mask of shape [1, seq_len] where 1 = suppress, 0 = keep.
        """
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = tokens.input_ids[0]

        # Get token IDs for harmful words
        harmful_token_set = set()
        for word in harmful_words:
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            harmful_token_set.update(word_tokens)

        # Create mask
        mask = torch.zeros(1, len(input_ids), 1, device=device)
        for i, token_id in enumerate(input_ids):
            if token_id.item() in harmful_token_set:
                mask[0, i, 0] = 1.0

        return mask

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
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        # === SHGD parameters ===
        harmful_concepts: Optional[List[str]] = None,
        anchor_concepts: Optional[List[str]] = None,
        # Guidance strength
        anchor_guidance_scale: float = 3.0,
        # Phase boundaries (as fraction of total timesteps, 0=end, 1=start)
        guide_start_frac: float = 0.8,   # Phase 2 starts (high noise)
        guide_end_frac: float = 0.4,     # Phase 2 ends, Phase 3 starts
        # Healing parameters
        heal_strength: float = 0.3,      # Re-noise amount (fraction of remaining noise)
        heal_steps: Optional[int] = None, # Override: exact number of re-denoise steps
        # Self-consistency monitor
        enable_self_consistency: bool = True,
        consistency_threshold: float = 0.85,
        consistency_check_interval: int = 5,
        # Adaptive healing
        adaptive_heal: bool = True,
        min_heal_strength: float = 0.1,
        max_heal_strength: float = 0.5,
        # Multi-round micro-healing
        micro_heal: bool = False,
        micro_heal_noise_scale: float = 0.05,
        # Frequency-selective healing
        freq_selective: bool = False,
        freq_cutoff: float = 0.3,  # fraction of spectrum to preserve (low-freq)
        # === v2: NEW FEATURES ===
        # Negative concept amplification
        negative_concept_scale: float = 0.0,  # Additional negative guidance from harmful concepts
        # Sample-adaptive guidance
        enable_sample_adaptive: bool = False,
        adaptive_guidance_min: float = 1.0,
        adaptive_guidance_max: float = 20.0,
        sample_adaptive_check_interval: int = 3,
        # Multi-round guide-heal
        num_guide_heal_rounds: int = 1,
        round_guidance_decay: float = 0.7,  # Each round reduces guidance by this factor
        # Progressive guidance schedule
        guidance_schedule: str = "constant",  # "constant", "cosine", "linear_decay", "warmup_decay"
        # Cross-attention manipulation
        enable_attn_suppression: bool = False,
        attn_suppress_scale: float = 0.1,  # Scale factor for harmful token attention (0=full suppress)
        harmful_words: Optional[List[str]] = None,  # Words to suppress in attention
        # Concept negation (use harmful as negative prompt too)
        enable_concept_negation: bool = False,
        concept_negation_scale: float = 3.0,
        # Return intermediate info
        return_dict: bool = True,
        return_latents: bool = False,
    ):
        # 0. Default handling
        if harmful_concepts is None:
            harmful_concepts = [
                "Sexual Acts", "Nudity", "Pornography",
                "Erotic Art", "Lustful", "Seductive",
            ]
        if anchor_concepts is None:
            anchor_concepts = [
                "a person wearing clothes", "a safe and appropriate scene",
                "a family-friendly image", "a fully dressed person",
            ]
        if harmful_words is None:
            harmful_words = [
                "nude", "naked", "nudity", "sex", "sexual", "erotic",
                "porn", "nsfw", "breast", "nipple", "genital", "penis",
                "vagina", "buttock", "ass", "boob", "topless", "strip",
                "undress", "bare", "flesh", "body", "skin", "exposed",
            ]

        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode text prompts
        # Main prompt embeddings (uncond + cond for standard CFG)
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt,
        )

        # Harmful concept embeddings (single forward, no CFG splitting)
        harmful_text = ", ".join(harmful_concepts)
        harmful_input_ids = self.tokenizer(
            harmful_text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        harmful_embeddings = self.text_encoder(harmful_input_ids)[0]
        harmful_embeddings = harmful_embeddings.repeat(
            batch_size * num_images_per_prompt, 1, 1,
        )

        # Anchor concept embeddings (single forward, no CFG splitting)
        anchor_text = ", ".join(anchor_concepts)
        anchor_input_ids = self.tokenizer(
            anchor_text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        anchor_embeddings = self.text_encoder(anchor_input_ids)[0]
        anchor_embeddings = anchor_embeddings.repeat(
            batch_size * num_images_per_prompt, 1, 1,
        )

        # For concept negation: create text_embeddings with harmful as negative
        if enable_concept_negation:
            # Replace the unconditional embeddings with harmful embeddings
            # This makes CFG push AWAY from harmful concepts
            if do_classifier_free_guidance:
                uncond_emb, cond_emb = text_embeddings.chunk(2)
                # Blend harmful into unconditional: uncond + scale * harmful direction
                negation_embeddings = torch.cat([
                    uncond_emb + concept_negation_scale * (harmful_embeddings - uncond_emb),
                    cond_emb,
                ])

        # Cross-attention: get harmful token mask for the prompt
        attn_token_mask = None
        if enable_attn_suppression:
            prompt_text = prompt if isinstance(prompt, str) else prompt[0]
            attn_token_mask = self._get_harmful_token_mask(
                prompt_text, harmful_words, device,
            )

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            text_embeddings.dtype, device, generator, latents,
        )

        # 4. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. Multi-round guide-heal loop
        current_anchor_scale = anchor_guidance_scale

        for round_idx in range(num_guide_heal_rounds):
            if round_idx > 0:
                # Decay guidance for subsequent rounds
                current_anchor_scale *= round_guidance_decay
                logger.info(
                    f"[SHGD] Round {round_idx + 1}/{num_guide_heal_rounds}, "
                    f"anchor_scale={current_anchor_scale:.2f}"
                )

            # Compute phase boundaries
            total_steps = len(timesteps)
            guide_start_step = int(total_steps * (1 - guide_start_frac))
            guide_end_step = int(total_steps * (1 - guide_end_frac))

            # For rounds > 0, we need to re-setup the scheduler
            if round_idx > 0:
                # Already have latents from previous round
                # Adjust phase boundaries for later rounds (shorter guide, lighter heal)
                guide_start_step = max(0, guide_start_step + int(total_steps * 0.1 * round_idx))
                guide_end_step = min(total_steps, guide_end_step + int(total_steps * 0.1 * round_idx))

            # 6. Denoising loop with SHGD
            consistency_scores = []
            phase_log = []
            healed = False

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

            # For first round, start from beginning; for later rounds, skip to guide phase
            start_step = 0 if round_idx == 0 else guide_start_step

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if i < start_step:
                        if round_idx == 0:
                            # Progress bar for skipped steps
                            pass
                        continue

                    if healed:
                        break

                    # Determine current phase
                    if i < guide_start_step:
                        phase = "monitor"
                    elif i < guide_end_step:
                        phase = "guide"
                    else:
                        phase = "heal"

                    # Compute scheduled guidance strength
                    if phase == "guide":
                        step_in_guide = i - guide_start_step
                        total_guide_steps = guide_end_step - guide_start_step
                        progress_in_guide = step_in_guide / max(total_guide_steps, 1)

                        scheduled_anchor_scale = self._compute_guidance_schedule(
                            current_anchor_scale, progress_in_guide, guidance_schedule,
                        )

                        # Sample-adaptive: scale based on current x0 prediction
                        if enable_sample_adaptive and step_in_guide % sample_adaptive_check_interval == 0:
                            harm_score = self._estimate_harm_score(
                                latents, t, text_embeddings,
                                harmful_embeddings, anchor_embeddings,
                                do_classifier_free_guidance, guidance_scale,
                            )
                            # Scale guidance proportionally to harm score
                            adaptive_factor = adaptive_guidance_min + harm_score * (
                                adaptive_guidance_max - adaptive_guidance_min
                            )
                            scheduled_anchor_scale = max(
                                scheduled_anchor_scale,
                                adaptive_factor,
                            )
                    else:
                        scheduled_anchor_scale = current_anchor_scale

                    # --- PHASE 1: MONITOR (standard CFG) ---
                    if phase == "monitor":
                        if enable_concept_negation:
                            # Use concept negation during monitor phase too
                            latents = self._standard_step(
                                latents, t, negation_embeddings,
                                do_classifier_free_guidance, guidance_scale,
                                extra_step_kwargs,
                            )
                        else:
                            latents = self._standard_step(
                                latents, t, text_embeddings,
                                do_classifier_free_guidance, guidance_scale,
                                extra_step_kwargs,
                            )

                    # --- PHASE 2: GUIDE (Dual-Anchor CFG + enhancements) ---
                    elif phase == "guide":
                        # Choose which text embeddings to use for base CFG
                        base_text_emb = negation_embeddings if enable_concept_negation else text_embeddings

                        latents, x0_hat = self._dual_anchor_step(
                            latents, t, base_text_emb,
                            harmful_embeddings, anchor_embeddings,
                            do_classifier_free_guidance, guidance_scale,
                            scheduled_anchor_scale, extra_step_kwargs,
                            negative_concept_scale=negative_concept_scale,
                            attn_token_mask=attn_token_mask,
                            attn_suppress_scale=attn_suppress_scale,
                            enable_attn_suppression=enable_attn_suppression,
                        )

                        # Micro-healing
                        if micro_heal:
                            latents = self._micro_heal_step(
                                latents, t, text_embeddings,
                                do_classifier_free_guidance, guidance_scale,
                                micro_heal_noise_scale, extra_step_kwargs,
                                generator,
                            )

                        # Self-consistency check
                        if enable_self_consistency and i % consistency_check_interval == 0:
                            score = self._compute_self_consistency(
                                latents, t, text_embeddings,
                                do_classifier_free_guidance, guidance_scale,
                            )
                            consistency_scores.append((i, t.item(), score))

                            if score < consistency_threshold:
                                logger.info(
                                    f"[SHGD] Off-manifold detected at step {i} "
                                    f"(t={t.item()}, consistency={score:.4f}). "
                                    f"Triggering early healing."
                                )
                                if adaptive_heal:
                                    actual_heal_strength = min(
                                        max_heal_strength,
                                        max(min_heal_strength,
                                            (1 - score) * max_heal_strength)
                                    )
                                else:
                                    actual_heal_strength = heal_strength

                                latents = self._heal(
                                    latents, i, timesteps,
                                    text_embeddings, do_classifier_free_guidance,
                                    guidance_scale, actual_heal_strength,
                                    heal_steps, extra_step_kwargs, generator,
                                    freq_selective, freq_cutoff,
                                )
                                healed = True
                                break

                    # --- PHASE 3: HEAL ---
                    elif phase == "heal" and not healed:
                        if adaptive_heal and consistency_scores:
                            last_score = consistency_scores[-1][2]
                            actual_heal_strength = min(
                                max_heal_strength,
                                max(min_heal_strength,
                                    (1 - last_score) * max_heal_strength)
                            )
                        else:
                            actual_heal_strength = heal_strength

                        latents = self._heal(
                            latents, i, timesteps,
                            text_embeddings, do_classifier_free_guidance,
                            guidance_scale, actual_heal_strength,
                            heal_steps, extra_step_kwargs, generator,
                            freq_selective, freq_cutoff,
                        )
                        healed = True
                        break

                    phase_log.append((i, t.item(), phase))

                    # Progress bar
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and
                        (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

            # After round completes, if more rounds remain, re-noise slightly
            if round_idx < num_guide_heal_rounds - 1 and healed:
                # Light re-noise for next round
                re_noise_strength = 0.15 * (1 - round_idx / num_guide_heal_rounds)
                re_noise_step = int(total_steps * re_noise_strength)
                re_noise_t = timesteps[re_noise_step]

                # Predict x0
                latent_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_input = self.scheduler.scale_model_input(latent_input, timesteps[-1])
                noise_pred = self.unet(
                    latent_input, timesteps[-1],
                    encoder_hidden_states=text_embeddings,
                ).sample
                if do_classifier_free_guidance:
                    n_u, n_t = noise_pred.chunk(2)
                    noise_pred = n_u + guidance_scale * (n_t - n_u)

                # Tweedie
                alpha_t = self.scheduler.alphas_cumprod[timesteps[-1]]
                x0_hat = (latents - (1 - alpha_t) ** 0.5 * noise_pred) / alpha_t ** 0.5

                # Re-noise
                noise = torch.randn(x0_hat.shape, generator=generator,
                                    device=device, dtype=x0_hat.dtype)
                latents = self.scheduler.add_noise(x0_hat, noise, re_noise_t)

                # Reset for next round
                healed = False

        if return_latents:
            return latents

        # 7. Decode
        image = self.decode_latents(latents)

        # 8. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image

    def _compute_guidance_schedule(self, base_scale, progress, schedule_type):
        """
        Compute guidance scale at a given progress point in the guide phase.

        Args:
            base_scale: Base anchor guidance scale
            progress: 0.0 (start of guide phase) to 1.0 (end of guide phase)
            schedule_type: "constant", "cosine", "linear_decay", "warmup_decay"
        """
        if schedule_type == "constant":
            return base_scale
        elif schedule_type == "cosine":
            # Cosine decay: starts at base_scale, decays to ~0
            return base_scale * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == "linear_decay":
            # Linear decay from base_scale to base_scale * 0.2
            return base_scale * (1.0 - 0.8 * progress)
        elif schedule_type == "warmup_decay":
            # Warmup to peak then decay
            if progress < 0.3:
                return base_scale * (progress / 0.3)
            else:
                return base_scale * (1.0 - 0.8 * (progress - 0.3) / 0.7)
        elif schedule_type == "bell":
            # Bell curve: peaks in the middle
            return base_scale * math.exp(-8 * (progress - 0.5) ** 2)
        else:
            return base_scale

    def _estimate_harm_score(self, latents, t, text_embeddings,
                             harmful_embeddings, anchor_embeddings,
                             do_cfg, guidance_scale):
        """
        Estimate how "harmful" the current sample is based on x0 prediction.

        Uses cosine similarity between x0_hat features and harmful/anchor concepts.
        Returns a score in [0, 1] where 1 = very harmful, 0 = safe.
        """
        # Predict x0 from current z_t
        latent_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_input = self.scheduler.scale_model_input(latent_input, t)
        noise_pred = self.unet(
            latent_input, t, encoder_hidden_states=text_embeddings,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Tweedie x0 estimate
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        x0_hat = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # Get noise predictions conditioned on harmful vs anchor concepts
        latent_input_single = self.scheduler.scale_model_input(latents, t)

        noise_harm = self.unet(
            latent_input_single, t, encoder_hidden_states=harmful_embeddings,
        ).sample
        noise_anchor = self.unet(
            latent_input_single, t, encoder_hidden_states=anchor_embeddings,
        ).sample

        # Compute similarity: how much does the current noise pred align with harmful vs anchor?
        noise_flat = noise_pred.reshape(1, -1)
        harm_flat = noise_harm.reshape(1, -1)
        anchor_flat = noise_anchor.reshape(1, -1)

        sim_harm = F.cosine_similarity(noise_flat, harm_flat, dim=-1).item()
        sim_anchor = F.cosine_similarity(noise_flat, anchor_flat, dim=-1).item()

        # Normalize to [0, 1]: higher = more harmful
        # If noise pred is more similar to harmful than anchor, score is higher
        harm_score = max(0.0, min(1.0, (sim_harm - sim_anchor + 1) / 2))

        return harm_score

    def _standard_step(self, latents, t, text_embeddings,
                       do_cfg, guidance_scale, extra_step_kwargs):
        """Standard DDPM/DDIM step with CFG."""
        latent_input = (
            torch.cat([latents] * 2) if do_cfg else latents
        )
        latent_input = self.scheduler.scale_model_input(latent_input, t)
        noise_pred = self.unet(
            latent_input, t, encoder_hidden_states=text_embeddings,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = self.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs,
        ).prev_sample
        return latents

    def _dual_anchor_step(self, latents, t, text_embeddings,
                          harmful_embeddings, anchor_embeddings,
                          do_cfg, guidance_scale, anchor_guidance_scale,
                          extra_step_kwargs,
                          negative_concept_scale=0.0,
                          attn_token_mask=None,
                          attn_suppress_scale=0.1,
                          enable_attn_suppression=False):
        """
        Enhanced Dual-Anchor CFG step:

        Base: e_hat = e_uncond + s*(e_cond - e_uncond) + alpha*(e_anchor - e_harm)
        + Negative amplification: - beta * e_harm
        + Attention suppression: modify text embeddings to suppress harmful tokens

        Returns (latents, x0_hat)
        """
        # Apply attention suppression to text embeddings if enabled
        if enable_attn_suppression and attn_token_mask is not None:
            # Suppress harmful tokens in the conditional embeddings
            if do_cfg:
                uncond_emb, cond_emb = text_embeddings.chunk(2)
                # Scale down harmful token embeddings
                suppressed_cond = cond_emb * (1 - attn_token_mask * (1 - attn_suppress_scale))
                text_embeddings_modified = torch.cat([uncond_emb, suppressed_cond])
            else:
                text_embeddings_modified = text_embeddings * (
                    1 - attn_token_mask * (1 - attn_suppress_scale)
                )
        else:
            text_embeddings_modified = text_embeddings

        # Standard CFG forward pass
        latent_input = (
            torch.cat([latents] * 2) if do_cfg else latents
        )
        latent_input_std = self.scheduler.scale_model_input(latent_input, t)
        noise_pred_std = self.unet(
            latent_input_std, t, encoder_hidden_states=text_embeddings_modified,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred_std.chunk(2)
            noise_pred_cfg = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        else:
            noise_pred_cfg = noise_pred_std

        # Harmful concept forward pass
        latent_input_single = self.scheduler.scale_model_input(latents, t)
        noise_pred_harm = self.unet(
            latent_input_single, t, encoder_hidden_states=harmful_embeddings,
        ).sample

        # Anchor concept forward pass
        noise_pred_anchor = self.unet(
            latent_input_single, t, encoder_hidden_states=anchor_embeddings,
        ).sample

        # Combined guidance:
        # Base dual-anchor + negative amplification
        noise_pred = noise_pred_cfg + anchor_guidance_scale * (
            noise_pred_anchor - noise_pred_harm
        )

        # Negative concept amplification: push away from harmful
        if negative_concept_scale > 0:
            noise_pred = noise_pred - negative_concept_scale * (
                noise_pred_harm - noise_uncond if do_cfg else noise_pred_harm
            )

        # Get x0_hat for consistency monitoring
        step_output = self.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs,
        )
        x0_hat = step_output.pred_original_sample
        latents = step_output.prev_sample

        return latents, x0_hat

    def _compute_self_consistency(self, latents, t, text_embeddings,
                                  do_cfg, guidance_scale):
        """
        Compute self-consistency score to detect off-manifold latents.

        Process:
        1. Predict x0 from current z_t
        2. Re-noise x0 back to z_t
        3. Re-predict x0' from reconstructed z_t
        4. Measure cosine similarity between x0 and x0'

        Low similarity -> off-manifold -> need healing
        """
        # Step 1: Predict x0 from z_t
        latent_input = (
            torch.cat([latents] * 2) if do_cfg else latents
        )
        latent_input = self.scheduler.scale_model_input(latent_input, t)
        noise_pred = self.unet(
            latent_input, t, encoder_hidden_states=text_embeddings,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Tweedie estimate of x0
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        x0_hat = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # Step 2: Re-noise x0 to z_t
        noise = torch.randn_like(x0_hat)
        z_t_recon = (
            alpha_prod_t ** 0.5 * x0_hat + beta_prod_t ** 0.5 * noise
        )

        # Step 3: Re-predict x0' from z_t_recon
        latent_input2 = (
            torch.cat([z_t_recon] * 2) if do_cfg else z_t_recon
        )
        latent_input2 = self.scheduler.scale_model_input(latent_input2, t)
        noise_pred2 = self.unet(
            latent_input2, t, encoder_hidden_states=text_embeddings,
        ).sample

        if do_cfg:
            noise_uncond2, noise_text2 = noise_pred2.chunk(2)
            noise_pred2 = noise_uncond2 + guidance_scale * (noise_text2 - noise_uncond2)

        x0_hat2 = (z_t_recon - beta_prod_t ** 0.5 * noise_pred2) / alpha_prod_t ** 0.5

        # Step 4: Cosine similarity
        x0_flat = x0_hat.reshape(x0_hat.shape[0], -1)
        x0_flat2 = x0_hat2.reshape(x0_hat2.shape[0], -1)
        score = F.cosine_similarity(x0_flat, x0_flat2, dim=-1).mean().item()

        return score

    def _heal(self, latents, current_step_idx, timesteps,
              text_embeddings, do_cfg, guidance_scale,
              heal_strength, heal_steps, extra_step_kwargs,
              generator, freq_selective, freq_cutoff):
        """
        Healing phase: re-noise + re-denoise with standard CFG only.

        The diffusion prior repairs artifacts from strong guidance
        while preserving the semantic shift (harmful -> safe).
        """
        remaining_timesteps = timesteps[current_step_idx:]
        n_remaining = len(remaining_timesteps)

        if n_remaining == 0:
            return latents

        # Compute how many steps to jump back (re-noise amount)
        jump_back = max(1, int(n_remaining * heal_strength))
        if heal_steps is not None:
            jump_back = min(heal_steps, n_remaining - 1)

        # Target timestep to re-noise to
        heal_target_idx = max(0, current_step_idx - jump_back)
        heal_target_t = timesteps[heal_target_idx]

        logger.info(
            f"[SHGD] Healing: re-noising from step {current_step_idx} "
            f"(t={timesteps[current_step_idx].item()}) back to step "
            f"{heal_target_idx} (t={heal_target_t.item()}), "
            f"then re-denoising {len(timesteps) - heal_target_idx} steps"
        )

        # First, predict x0_hat from current noisy latent (Tweedie estimate)
        latent_input = (
            torch.cat([latents] * 2) if do_cfg else latents
        )
        latent_input = self.scheduler.scale_model_input(
            latent_input, timesteps[current_step_idx]
        )
        noise_pred = self.unet(
            latent_input, timesteps[current_step_idx],
            encoder_hidden_states=text_embeddings,
        ).sample
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        x0_hat = self.scheduler.step(
            noise_pred, timesteps[current_step_idx], latents,
            **extra_step_kwargs,
        ).pred_original_sample

        # Re-noise x0_hat to heal_target_t
        noise = torch.randn(
            x0_hat.shape, generator=generator,
            device=x0_hat.device, dtype=x0_hat.dtype,
        )

        if freq_selective:
            latents = self._freq_selective_renoise(
                x0_hat, heal_target_t, freq_cutoff, generator,
            )
        else:
            latents = self.scheduler.add_noise(x0_hat, noise, heal_target_t)

        # Re-denoise from heal_target to end with standard CFG only
        heal_timesteps = timesteps[heal_target_idx:]
        for j, t_heal in enumerate(heal_timesteps):
            latents = self._standard_step(
                latents, t_heal, text_embeddings,
                do_cfg, guidance_scale, extra_step_kwargs,
            )

        return latents

    def _micro_heal_step(self, latents, t, text_embeddings,
                         do_cfg, guidance_scale, noise_scale,
                         extra_step_kwargs, generator):
        """
        Micro-healing: after each guided step, add small noise and
        do one standard step to smooth out artifacts.
        """
        noise = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )
        latents = latents + noise_scale * noise

        latent_input = (
            torch.cat([latents] * 2) if do_cfg else latents
        )
        latent_input = self.scheduler.scale_model_input(latent_input, t)
        noise_pred = self.unet(
            latent_input, t, encoder_hidden_states=text_embeddings,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Small correction toward manifold
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        correction = -noise_scale * (1 - alpha_prod_t) ** 0.5 * noise_pred
        latents = latents + 0.5 * correction

        return latents

    def _freq_selective_renoise(self, latents, target_t, freq_cutoff,
                                generator):
        """
        Frequency-selective re-noising:
        - Preserve low-frequency components (semantic content / concept shift)
        - Re-noise high-frequency components (artifacts)
        """
        B, C, H, W = latents.shape

        freq = torch.fft.fft2(latents)
        freq_shifted = torch.fft.fftshift(freq)

        cy, cx = H // 2, W // 2
        radius = int(min(H, W) * freq_cutoff / 2)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=latents.device),
            torch.arange(W, device=latents.device),
            indexing='ij',
        )
        mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius ** 2
        mask = mask.float().unsqueeze(0).unsqueeze(0)

        low_freq = freq_shifted * mask
        low_latent = torch.fft.ifft2(torch.fft.ifftshift(low_freq)).real

        noise = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )
        noise_freq = torch.fft.fftshift(torch.fft.fft2(noise))
        noise_high = noise_freq * (1 - mask)
        noise_high_spatial = torch.fft.ifft2(
            torch.fft.ifftshift(noise_high)
        ).real

        alpha_prod_t = self.scheduler.alphas_cumprod[target_t]
        latents = low_latent + (1 - alpha_prod_t) ** 0.5 * noise_high_spatial

        return latents
