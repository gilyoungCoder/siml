"""
Flow matching denoising utilities for REPA/SiT.

Replaces the DDPM-based denoise_utils.py from z0_clf_guidance.
Key difference: linear interpolation path instead of DDPM noise schedule.

Forward:  x_t = (1-t) * x_0 + t * noise,   t in [0,1]
Model predicts velocity:  v = -x_0 + noise  (for linear path)
One-step denoising:  x_0 = x_t - t * v
"""

import torch


def predict_x0_from_velocity(x_t, v_pred, t):
    """
    One-step denoising for flow matching (linear path).

    x_0_hat = x_t - t * v_pred

    Note: d(x0_hat)/d(x_t) = I (identity), unlike DDPM's 1/sqrt(alpha_bar).
    This means classifier guidance gradients are uniform across all timesteps.

    Args:
        x_t: noisy latent (B, C, H, W)
        v_pred: predicted velocity (B, C, H, W)
        t: scalar float or (B,) tensor, timestep in [0, 1]

    Returns:
        x0_hat: predicted clean latent (B, C, H, W)
    """
    if isinstance(t, (int, float)):
        t_expanded = t
    else:
        t_expanded = t.view(-1, 1, 1, 1)
    return x_t - t_expanded * v_pred


def inject_noise_flow(x_0, noise, t):
    """
    Forward noising for flow matching (linear path).

    x_t = (1 - t) * x_0 + t * noise

    Args:
        x_0: clean latent (B, C, H, W)
        noise: random noise (B, C, H, W)
        t: (B, 1, 1, 1) timestep values in [0, 1]

    Returns:
        x_t: noisy latent (B, C, H, W)
    """
    return (1 - t) * x_0 + t * noise
