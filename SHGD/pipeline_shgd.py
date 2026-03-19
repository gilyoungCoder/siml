"""
Self-Healing Guided Diffusion (SHGD) Pipeline

Training-free safe generation via:
1. Dual-Anchor CFG in critical time window (inspired by SGF's CBF proof):
   guidance only in early denoising steps [1.0, ~0.78] where semantic layout is decided
2. Strong guidance + heal: push hard from harmful -> safe, then heal quality
3. Self-Consistency Monitor: detect off-manifold and trigger early healing
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
    SHGD: Strong guidance in critical window + heal.

    Key insight from SGF (ICLR 2026): guidance is only effective in the
    critical time window [1.0, ~0.78] (early denoising). After that,
    guidance should be zero. We apply very strong dual-anchor CFG in this
    window, then heal any quality degradation via SDEdit-style re-noising.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        callback=None,
        callback_steps=1,
        # === SHGD parameters ===
        harmful_concepts=None,
        anchor_concepts=None,
        # Guidance strength (strong!)
        anchor_guidance_scale=10.0,
        # Critical time window (SGF-inspired)
        # guide_start_frac=1.0 means guidance starts from the very first step
        # guide_end_frac=0.78 means guidance stops at ~78% of total noise schedule
        guide_start_frac=1.0,
        guide_end_frac=0.78,
        # Healing parameters
        heal_strength=0.4,
        heal_steps=None,
        # Self-consistency monitor
        enable_self_consistency=True,
        consistency_threshold=0.85,
        consistency_check_interval=3,
        # Adaptive healing
        adaptive_heal=True,
        min_heal_strength=0.2,
        max_heal_strength=0.6,
        # Micro-healing (per-step correction during guide)
        micro_heal=False,
        micro_heal_noise_scale=0.05,
        # Frequency-selective healing
        freq_selective=False,
        freq_cutoff=0.3,
        # Content-based trigger: only apply guidance if prompt is similar to harmful concepts
        enable_trigger=True,
        trigger_sim_threshold=0.32,
        # Return
        return_dict=True,
        return_latents=False,
    ):
        # 0. Defaults
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

        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_cfg = guidance_scale > 1.0

        # 1. Encode embeddings
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt,
            do_cfg, negative_prompt,
        )

        # Extract conditional embedding for safe heal step
        if do_cfg:
            cond_emb = text_embeddings.chunk(2)[1]  # [uncond, cond] -> cond
        else:
            cond_emb = text_embeddings

        harmful_text = ", ".join(harmful_concepts)
        harmful_ids = self.tokenizer(
            harmful_text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        harmful_emb = self.text_encoder(harmful_ids)[0].repeat(
            batch_size * num_images_per_prompt, 1, 1,
        )

        anchor_text = ", ".join(anchor_concepts)
        anchor_ids = self.tokenizer(
            anchor_text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        anchor_emb = self.text_encoder(anchor_ids)[0].repeat(
            batch_size * num_images_per_prompt, 1, 1,
        )

        # 2. Prepare timesteps & latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        total_steps = len(timesteps)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels, height, width,
            text_embeddings.dtype, device, generator, latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 3. Phase boundaries
        # guide_start_frac=1.0 -> guide starts at step 0 (highest noise)
        # guide_end_frac=0.78 -> guide ends at step ~11 (for 50 steps)
        guide_start_step = int(total_steps * (1 - guide_start_frac))
        guide_end_step = int(total_steps * (1 - guide_end_frac))

        logger.info(
            f"[SHGD] Critical window: steps {guide_start_step}-{guide_end_step} "
            f"(frac [{guide_start_frac}, {guide_end_frac}]), "
            f"anchor_scale={anchor_guidance_scale}, "
            f"heal_strength={heal_strength}"
        )

        # 4. Denoising loop
        consistency_scores = []
        healed = False
        guidance_triggered = False  # Whether content-based trigger fired
        trigger_checked = False     # Whether we've done the trigger check
        num_warmup = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                if healed:
                    break

                # Phase decision
                in_guide_window = guide_start_step <= i < guide_end_step
                past_guide_window = i >= guide_end_step

                # Content-based trigger check at first guided step
                if in_guide_window and not trigger_checked and enable_trigger:
                    trigger_checked = True
                    trigger_sim = self._check_trigger(cond_emb, harmful_emb)
                    guidance_triggered = trigger_sim > trigger_sim_threshold
                    logger.info(
                        f"[SHGD] Trigger check: sim={trigger_sim:.4f}, "
                        f"threshold={trigger_sim_threshold}, "
                        f"triggered={guidance_triggered}"
                    )
                    if not guidance_triggered:
                        # Safe prompt — skip all guidance, run standard for rest
                        in_guide_window = False

                # If trigger is enabled but not triggered, skip guidance
                if enable_trigger and trigger_checked and not guidance_triggered:
                    in_guide_window = False
                    past_guide_window = False

                if in_guide_window:
                    # === GUIDE: strong dual-anchor CFG ===
                    latents = self._dual_anchor_step(
                        latents, t, text_embeddings,
                        harmful_emb, anchor_emb,
                        do_cfg, guidance_scale,
                        anchor_guidance_scale, extra_step_kwargs,
                    )

                    if micro_heal:
                        latents = self._micro_heal_step(
                            latents, t, text_embeddings,
                            do_cfg, guidance_scale,
                            micro_heal_noise_scale, extra_step_kwargs,
                            generator,
                        )

                    # Self-consistency check for early healing
                    if enable_self_consistency and i % consistency_check_interval == 0:
                        score = self._compute_self_consistency(
                            latents, t, text_embeddings, do_cfg, guidance_scale,
                        )
                        consistency_scores.append((i, t.item(), score))

                        if score < consistency_threshold:
                            logger.info(
                                f"[SHGD] Off-manifold at step {i} "
                                f"(consistency={score:.4f}). Early heal."
                            )
                            actual_hs = self._get_heal_strength(
                                adaptive_heal, consistency_scores,
                                heal_strength, min_heal_strength, max_heal_strength,
                            )
                            latents = self._heal(
                                latents, i, timesteps, text_embeddings,
                                do_cfg, guidance_scale, actual_hs,
                                heal_steps, extra_step_kwargs, generator,
                                freq_selective, freq_cutoff,
                                harm_emb=harmful_emb, cond_emb=cond_emb,
                            )
                            healed = True
                            break

                elif past_guide_window and not healed:
                    # === HEAL: transition from guided to standard ===
                    actual_hs = self._get_heal_strength(
                        adaptive_heal, consistency_scores,
                        heal_strength, min_heal_strength, max_heal_strength,
                    )
                    latents = self._heal(
                        latents, i, timesteps, text_embeddings,
                        do_cfg, guidance_scale, actual_hs,
                        heal_steps, extra_step_kwargs, generator,
                        freq_selective, freq_cutoff,
                        harm_emb=harmful_emb, cond_emb=cond_emb,
                    )
                    healed = True
                    break

                else:
                    # Before guide window (if guide_start_frac < 1.0)
                    latents = self._standard_step(
                        latents, t, text_embeddings,
                        do_cfg, guidance_scale, extra_step_kwargs,
                    )

                # Progress
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup and
                    (i + 1) % self.scheduler.order == 0
                ):
                    pbar.update()
                    if callback and i % callback_steps == 0:
                        callback(i, t, latents)

        if return_latents:
            return latents

        image = self.decode_latents(latents)
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image

    def _get_heal_strength(self, adaptive, scores, base, min_hs, max_hs):
        if adaptive and scores:
            last_score = scores[-1][2]
            return min(max_hs, max(min_hs, (1 - last_score) * max_hs))
        return base

    def _check_trigger(self, prompt_emb, harm_emb):
        """
        CLIP text-space trigger: cosine similarity between mean-pooled
        prompt embedding and harmful concept embedding.
        Fast (no UNet call) and accurate (97.8% on calibration set).
        """
        # Mean-pool over token dimension [B, 77, 768] -> [B, 768]
        p = prompt_emb.mean(dim=1)
        h = harm_emb.mean(dim=1)
        return F.cosine_similarity(p, h, dim=-1).mean().item()

    def _standard_step(self, latents, t, text_emb, do_cfg, gs, extra):
        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = self.scheduler.scale_model_input(inp, t)
        pred = self.unet(inp, t, encoder_hidden_states=text_emb).sample
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + gs * (c - u)
        return self.scheduler.step(pred, t, latents, **extra).prev_sample

    def _dual_anchor_step(self, latents, t, text_emb, harm_emb, anchor_emb,
                          do_cfg, gs, anchor_gs, extra):
        """
        e_hat = e_uncond + s*(e_cond - e_uncond) + alpha*(e_anchor - e_harm)
        Strong alpha in critical window, then heal.
        """
        # Standard CFG
        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = self.scheduler.scale_model_input(inp, t)
        pred_std = self.unet(inp, t, encoder_hidden_states=text_emb).sample
        if do_cfg:
            u, c = pred_std.chunk(2)
            pred_cfg = u + gs * (c - u)
        else:
            pred_cfg = pred_std

        # Harmful & anchor
        inp_s = self.scheduler.scale_model_input(latents, t)
        pred_harm = self.unet(inp_s, t, encoder_hidden_states=harm_emb).sample
        pred_anchor = self.unet(inp_s, t, encoder_hidden_states=anchor_emb).sample

        # Combined: base CFG + strong directional shift
        pred = pred_cfg + anchor_gs * (pred_anchor - pred_harm)

        out = self.scheduler.step(pred, t, latents, **extra)
        return out.prev_sample

    def _compute_self_consistency(self, latents, t, text_emb, do_cfg, gs):
        """Predict x0, re-noise, re-predict. Low similarity = off-manifold."""
        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = self.scheduler.scale_model_input(inp, t)
        pred = self.unet(inp, t, encoder_hidden_states=text_emb).sample
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + gs * (c - u)

        a_t = self.scheduler.alphas_cumprod[t]
        b_t = 1 - a_t
        x0 = (latents - b_t ** 0.5 * pred) / a_t ** 0.5

        noise = torch.randn_like(x0)
        z_t = a_t ** 0.5 * x0 + b_t ** 0.5 * noise

        inp2 = torch.cat([z_t] * 2) if do_cfg else z_t
        inp2 = self.scheduler.scale_model_input(inp2, t)
        pred2 = self.unet(inp2, t, encoder_hidden_states=text_emb).sample
        if do_cfg:
            u2, c2 = pred2.chunk(2)
            pred2 = u2 + gs * (c2 - u2)
        x0_2 = (z_t - b_t ** 0.5 * pred2) / a_t ** 0.5

        return F.cosine_similarity(
            x0.reshape(x0.shape[0], -1),
            x0_2.reshape(x0_2.shape[0], -1),
            dim=-1,
        ).mean().item()

    def _safe_step(self, latents, t, cond_emb, harm_emb, gs, extra):
        """
        CFG with harmful embedding as negative prompt:
        ε̂ = ε_harm + gs * (ε_cond - ε_harm)
        This steers AWAY from harmful concepts while maintaining prompt fidelity.
        """
        inp_s = self.scheduler.scale_model_input(latents, t)
        pred_harm = self.unet(inp_s, t, encoder_hidden_states=harm_emb).sample
        pred_cond = self.unet(inp_s, t, encoder_hidden_states=cond_emb).sample
        pred = pred_harm + gs * (pred_cond - pred_harm)
        return self.scheduler.step(pred, t, latents, **extra).prev_sample

    def _heal(self, latents, step_idx, timesteps, text_emb,
              do_cfg, gs, heal_strength, heal_steps, extra,
              generator, freq_sel, freq_cutoff,
              harm_emb=None, cond_emb=None):
        """
        Re-noise x0_hat + re-denoise.
        If harm_emb is provided, uses safe CFG (harm as negative) to preserve
        guidance effect during healing. Otherwise falls back to standard CFG.
        """
        remaining = timesteps[step_idx:]
        n_rem = len(remaining)
        if n_rem == 0:
            return latents

        jump = max(1, int(n_rem * heal_strength))
        if heal_steps is not None:
            jump = min(heal_steps, n_rem - 1)

        target_idx = max(0, step_idx - jump)
        target_t = timesteps[target_idx]

        # Tweedie x0
        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = self.scheduler.scale_model_input(inp, timesteps[step_idx])
        pred = self.unet(inp, timesteps[step_idx], encoder_hidden_states=text_emb).sample
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + gs * (c - u)

        x0_hat = self.scheduler.step(
            pred, timesteps[step_idx], latents, **extra,
        ).pred_original_sample

        # Re-noise
        noise = torch.randn(x0_hat.shape, generator=generator,
                            device=x0_hat.device, dtype=x0_hat.dtype)

        if freq_sel:
            latents = self._freq_selective_renoise(
                x0_hat, target_t, freq_cutoff, generator,
            )
        else:
            latents = self.scheduler.add_noise(x0_hat, noise, target_t)

        # Re-denoise: use safe CFG if harmful embedding available
        if harm_emb is not None and cond_emb is not None:
            for t_h in timesteps[target_idx:]:
                latents = self._safe_step(
                    latents, t_h, cond_emb, harm_emb, gs, extra,
                )
        else:
            for t_h in timesteps[target_idx:]:
                latents = self._standard_step(
                    latents, t_h, text_emb, do_cfg, gs, extra,
                )

        return latents

    def _micro_heal_step(self, latents, t, text_emb, do_cfg, gs,
                         noise_scale, extra, generator):
        noise = torch.randn(latents.shape, generator=generator,
                            device=latents.device, dtype=latents.dtype)
        latents = latents + noise_scale * noise

        inp = torch.cat([latents] * 2) if do_cfg else latents
        inp = self.scheduler.scale_model_input(inp, t)
        pred = self.unet(inp, t, encoder_hidden_states=text_emb).sample
        if do_cfg:
            u, c = pred.chunk(2)
            pred = u + gs * (c - u)

        a_t = self.scheduler.alphas_cumprod[t]
        correction = -noise_scale * (1 - a_t) ** 0.5 * pred
        return latents + 0.5 * correction

    def _freq_selective_renoise(self, latents, target_t, freq_cutoff, generator):
        B, C, H, W = latents.shape
        freq = torch.fft.fftshift(torch.fft.fft2(latents))

        cy, cx = H // 2, W // 2
        r = int(min(H, W) * freq_cutoff / 2)
        yg, xg = torch.meshgrid(
            torch.arange(H, device=latents.device),
            torch.arange(W, device=latents.device),
            indexing='ij',
        )
        mask = (((yg - cy) ** 2 + (xg - cx) ** 2) <= r ** 2).float()[None, None]

        low = torch.fft.ifft2(torch.fft.ifftshift(freq * mask)).real

        noise = torch.randn(latents.shape, generator=generator,
                            device=latents.device, dtype=latents.dtype)
        nf = torch.fft.fftshift(torch.fft.fft2(noise))
        nh = torch.fft.ifft2(torch.fft.ifftshift(nf * (1 - mask))).real

        a_t = self.scheduler.alphas_cumprod[target_t]
        return low + (1 - a_t) ** 0.5 * nh
