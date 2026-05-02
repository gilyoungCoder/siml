import torch


def get_alpha_bar(scheduler, timesteps, device):
    """
    Retrieve alpha_bar (alpha_cumprod) for given timesteps.
    Returns shape: (B, 1, 1, 1) for broadcasting with latent tensors.
    """
    alpha_bar = scheduler.alphas_cumprod.to(device)[timesteps]
    return alpha_bar.view(-1, 1, 1, 1)


def predict_z0(zt, noise_pred, alpha_bar):
    """
    One-step denoising via Tweedie formula (DDIM-style x0 prediction).

    z0_hat = (zt - sqrt(1 - alpha_bar) * noise_pred) / sqrt(alpha_bar)

    Args:
        zt: noisy latent (B, 4, 64, 64)
        noise_pred: predicted noise (B, 4, 64, 64)
        alpha_bar: alpha_cumprod values (B, 1, 1, 1)

    Returns:
        z0_hat: predicted clean latent (B, 4, 64, 64)
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    z0_hat = (zt - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return z0_hat


def inject_noise(z0, noise, alpha_bar):
    """
    Forward diffusion: zt = sqrt(alpha_bar) * z0 + sqrt(1 - alpha_bar) * noise

    Args:
        z0: clean latent (B, 4, 64, 64)
        noise: random noise (B, 4, 64, 64)
        alpha_bar: alpha_cumprod values (B, 1, 1, 1)

    Returns:
        zt: noisy latent (B, 4, 64, 64)
    """
    return torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
