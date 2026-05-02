#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLASS Flows: Transition Sampling for SD1.4 (VP Schedule)

Implements Algorithm 1 from:
  "GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models"
  Holderrieth et al., ICLR 2026 (arxiv 2509.25170)

GLASS samples Markov transitions p_{t'|t}(x_{t'}|x_t) via an inner ODE,
combining ODE efficiency with SDE stochasticity. Uses the sufficient statistics
trick to reuse the pre-trained denoiser without any retraining.

Usage:
    sampler = GlassSampler(scheduler, unet, rho=0.4, inner_steps=4)
    # Single transition
    x_next = sampler.sample_transition(x_t, t_idx, t_prime_idx, cond_embeds)
    # Full trajectory (noise -> clean)
    z0 = sampler.sample_full_trajectory(z_T, timesteps, cond_embeds)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GlassSampler:
    """
    GLASS Flows transition sampler for VP diffusion (SD1.4).

    The core idea: construct an "inner flow matching model" that samples
    transitions p_{t'|t} using ODEs. Diversity is controlled by the
    correlation parameter rho.
    """

    def __init__(
        self,
        scheduler,
        unet,
        rho: float = 0.4,
        inner_steps: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            scheduler: DDIMScheduler with alphas_cumprod
            unet: SD1.4 UNet2DConditionModel
            rho: Correlation parameter (-1 to 1). Controls diversity.
                 0.4 = good diversity (GLASS paper default for FLUX)
                 DDPM-equivalent: rho = alpha_t * sigma_{t'} / (sigma_t * alpha_{t'})
            inner_steps: M, number of inner ODE steps per transition
            device: torch device
        """
        self.scheduler = scheduler
        self.unet = unet
        self.rho = rho
        self.inner_steps = inner_steps
        self.device = device or next(unet.parameters()).device

        # Cache alphas_cumprod as float64 for numerical stability
        self.alphas_cumprod = scheduler.alphas_cumprod.double().to(self.device)

    # -----------------------------------------------------------------
    # Schedule helpers (VP: alpha_t = sqrt(acp_t), sigma_t = sqrt(1-acp_t))
    # -----------------------------------------------------------------
    def _get_alpha_sigma(self, t_int: int):
        """Get alpha_t, sigma_t for a discrete timestep (integer)."""
        acp = self.alphas_cumprod[t_int].clamp(min=1e-12, max=1.0 - 1e-7)
        alpha = acp.sqrt()
        sigma = (1.0 - acp).sqrt()
        return alpha, sigma

    def _g(self, t_int: int):
        """g(t) = sigma_t^2 / alpha_t^2 = (1-acp)/acp (effective noise scale)."""
        acp = self.alphas_cumprod[t_int].clamp(min=1e-12, max=1.0 - 1e-7)
        return (1.0 - acp) / acp

    def _g_inv(self, y: torch.Tensor) -> int:
        """g^{-1}(y) -> closest discrete timestep. VP: acp_target = 1/(1+y)."""
        y = y.clamp(min=1e-12)
        acp_target = 1.0 / (1.0 + y)
        # Find closest timestep in alphas_cumprod
        diffs = (self.alphas_cumprod - acp_target).abs()
        t_star = diffs.argmin().item()
        return int(t_star)

    # -----------------------------------------------------------------
    # Denoiser: D(x_t, t) = (x_t - sigma_t * eps_theta) / alpha_t
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _denoiser(self, x_t, t_int: int, encoder_hidden_states):
        """Standard denoiser from eps-prediction."""
        alpha, sigma = self._get_alpha_sigma(t_int)
        # UNet expects fp16 input
        t_tensor = torch.tensor([t_int], device=self.device, dtype=torch.long)
        eps = self.unet(
            x_t.half(), t_tensor,
            encoder_hidden_states=encoder_hidden_states.half()
        ).sample.double()
        # D(x_t, t) = (x_t - sigma * eps) / alpha
        return (x_t - sigma * eps) / alpha.clamp(min=1e-8)

    # -----------------------------------------------------------------
    # GLASS Denoiser via sufficient statistics (Proposition 2)
    # -----------------------------------------------------------------
    def _glass_denoiser(
        self, x_t, x_bar_s, s: float,
        t_int: int, t_prime_int: int,
        gamma_bar, alpha_bar_s, sigma_bar_s,
        encoder_hidden_states,
    ):
        """
        GLASS denoiser: D_{mu,Sigma}(x_t, x_bar_s) = D(alpha_{t*} * S, t*)

        At s=0: falls back to D(x_t, t) (Markov property).
        """
        if s <= 1e-6:
            return self._denoiser(x_t, t_int, encoder_hidden_states)

        alpha_t, sigma_t = self._get_alpha_sigma(t_int)

        # Compute mu and Sigma (2x2 per coordinate)
        mu1 = alpha_t
        mu2 = alpha_bar_s + gamma_bar * alpha_t

        S11 = sigma_t ** 2
        S12 = sigma_t ** 2 * gamma_bar
        S22 = sigma_bar_s ** 2 + gamma_bar ** 2 * sigma_t ** 2

        # mu^T Sigma^{-1} mu (scalar)
        det = S11 * S22 - S12 * S12
        det = det.clamp(min=1e-12)
        Sinv11 = S22 / det
        Sinv12 = -S12 / det
        Sinv22 = S11 / det

        muSinvmu = mu1 * (Sinv11 * mu1 + Sinv12 * mu2) + \
                   mu2 * (Sinv12 * mu1 + Sinv22 * mu2)
        muSinvmu = muSinvmu.clamp(min=1e-12)

        # Sufficient statistic S(x) = mu^T Sigma^{-1} [x_t, x_bar_s]^T / (mu^T Sigma^{-1} mu)
        w_xt = mu1 * Sinv11 + mu2 * Sinv12
        w_xbar = mu1 * Sinv12 + mu2 * Sinv22
        S_val = (w_xt * x_t + w_xbar * x_bar_s) / muSinvmu

        # t* = g^{-1}((mu^T Sigma^{-1} mu)^{-1})
        t_star = self._g_inv(1.0 / muSinvmu)

        # D(alpha_{t*} * S, t*)
        alpha_tstar, _ = self._get_alpha_sigma(t_star)
        return self._denoiser(alpha_tstar * S_val, t_star, encoder_hidden_states)

    # -----------------------------------------------------------------
    # GLASS velocity field (Theorem 1)
    # -----------------------------------------------------------------
    def _glass_velocity(
        self, x_bar_s, x_t, s: float,
        t_int: int, t_prime_int: int,
        gamma_bar, alpha_bar_final, sigma_bar_final, sigma_bar_0,
        encoder_hidden_states,
    ):
        """
        u_s(x_bar_s | x_t, t) = w1 * x_bar_s + w2 * D_hat + w3 * x_t

        Uses CondOT inner schedulers:
            alpha_bar_s = s * alpha_bar_final
            sigma_bar_s = (1-s) * sigma_bar_0 + s * sigma_bar_final
        """
        # Inner schedule (CondOT)
        alpha_bar_s = s * alpha_bar_final
        sigma_bar_s = (1.0 - s) * sigma_bar_0 + s * sigma_bar_final

        # Derivatives
        d_alpha_bar_s = alpha_bar_final
        d_sigma_bar_s = sigma_bar_final - sigma_bar_0

        # Weight coefficients
        w1 = d_sigma_bar_s / sigma_bar_s.clamp(min=1e-8)
        w2 = d_alpha_bar_s - alpha_bar_s * w1
        w3 = -gamma_bar * w1

        # GLASS denoiser
        z_hat = self._glass_denoiser(
            x_t, x_bar_s, s, t_int, t_prime_int,
            gamma_bar, alpha_bar_s, sigma_bar_s,
            encoder_hidden_states,
        )

        return w1 * x_bar_s + w2 * z_hat + w3 * x_t

    # -----------------------------------------------------------------
    # Sample a single transition p_{t'|t}(x_{t'} | x_t)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_transition(
        self,
        x_t: torch.Tensor,
        t_int: int,
        t_prime_int: int,
        encoder_hidden_states: torch.Tensor,
        rho: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample x_{t'} ~ p_{t'|t}(. | x_t) using GLASS inner ODE.

        Args:
            x_t: current latent [B, 4, 64, 64]
            t_int: current discrete timestep (e.g., 981)
            t_prime_int: next discrete timestep (e.g., 961)
            encoder_hidden_states: text conditioning [B, 77, 768]
            rho: override correlation (default: self.rho)

        Returns:
            x_t_prime: [B, 4, 64, 64]
        """
        rho = rho if rho is not None else self.rho
        x_t = x_t.double()

        alpha_t, sigma_t = self._get_alpha_sigma(t_int)
        alpha_tp, sigma_tp = self._get_alpha_sigma(t_prime_int)

        # Inner schedule parameters (eq. 17)
        gamma_bar = rho * sigma_tp / sigma_t.clamp(min=1e-8)
        alpha_bar_final = alpha_tp - gamma_bar * alpha_t
        sigma_bar_final_sq = sigma_tp ** 2 * (1.0 - rho ** 2)
        sigma_bar_final = sigma_bar_final_sq.clamp(min=1e-12).sqrt()

        # Initial noise scale (sigma_bar_0 > 0 for well-defined inner ODE)
        sigma_bar_0 = sigma_bar_final  # CondOT: sigma_bar_0 = sigma_bar_final

        # Initialize inner trajectory: X_bar_0 = gamma_bar * x_t + sigma_bar_0 * eps
        eps = torch.randn_like(x_t)
        x_bar = gamma_bar * x_t + sigma_bar_0 * eps

        # Euler integration over inner time s: 0 -> 1
        M = self.inner_steps
        h = 1.0 / M
        for m in range(M):
            s = m * h
            # Avoid exact s=0 (use small positive value)
            s_safe = max(s, 1e-6)
            v = self._glass_velocity(
                x_bar, x_t, s_safe,
                t_int, t_prime_int,
                gamma_bar, alpha_bar_final, sigma_bar_final, sigma_bar_0,
                encoder_hidden_states,
            )
            x_bar = x_bar + h * v

        return x_bar.half()

    # -----------------------------------------------------------------
    # Full trajectory: z_T -> z_0 (generate a complete image)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_full_trajectory(
        self,
        z_T: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rho: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate a full image by chaining GLASS transitions.

        Args:
            z_T: initial noise [1, 4, 64, 64]
            timesteps: scheduler timesteps (e.g., [981, 961, ..., 1])
            encoder_hidden_states: text conditioning
            rho: override correlation

        Returns:
            z_0: denoised latent [1, 4, 64, 64]
        """
        x = z_T
        for i in range(len(timesteps) - 1):
            t_int = timesteps[i].item()
            t_prime_int = timesteps[i + 1].item()
            x = self.sample_transition(
                x, t_int, t_prime_int,
                encoder_hidden_states, rho=rho,
            )
        # Handle last step -> t=0
        if len(timesteps) > 0:
            last_t = timesteps[-1].item()
            x = self.sample_transition(
                x, last_t, 0,
                encoder_hidden_states, rho=rho,
            )
        return x
