"""
Flow matching denoising utilities for AuraFlow / Pony V7.

Forward:  x_t = (1 - sigma) * x_0 + sigma * noise,   sigma in [0, 1]
Model predicts velocity v such that: dx/dsigma = v
One-step denoising:  x_0_hat = x_t - sigma * v_pred
"""

import torch


def predict_x0_from_velocity(x_t, v_pred, sigma):
    """
    One-step denoising for flow matching (linear interpolation path).

    x_0_hat = x_t - sigma * v_pred

    Note: d(x0_hat)/d(x_t) = I (identity Jacobian), unlike DDPM's 1/sqrt(alpha_bar).
    This means classifier guidance gradients flow uniformly across all timesteps.

    Args:
        x_t: noisy latent (B, C, H, W)
        v_pred: predicted velocity (B, C, H, W)
        sigma: scalar float or (B,) or (B, 1, 1, 1) noise level in [0, 1]

    Returns:
        x0_hat: predicted clean latent (B, C, H, W)
    """
    if isinstance(sigma, (int, float)):
        sigma_expanded = sigma
    else:
        if sigma.dim() == 1:
            sigma_expanded = sigma.view(-1, 1, 1, 1)
        else:
            sigma_expanded = sigma
    return x_t - sigma_expanded * v_pred


def inject_noise_flow(x_0, noise, sigma):
    """
    Forward noising for flow matching (linear interpolation path).

    x_t = (1 - sigma) * x_0 + sigma * noise

    Args:
        x_0: clean latent (B, C, H, W)
        noise: random noise (B, C, H, W)
        sigma: (B, 1, 1, 1) noise level values in [0, 1]

    Returns:
        x_t: noisy latent (B, C, H, W)
    """
    return (1 - sigma) * x_0 + sigma * noise
