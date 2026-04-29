#!/usr/bin/env python3
"""FLUX-1 generation with **SAFREE + Safe-Denoiser repellency** (and optional SGF mode).

Algorithm (Safe-Denoiser, NeurIPS 2025, adapted to flow-matching FLUX):
  At each timestep t, get the predicted velocity v(z_t, t).
  Derive z_0 estimate:   z_0_hat = z_t - sigma_t * v
  Apply repellency:      z_0_hat <- z_0_hat - lambda(t) * sum_m w_m(z_0_hat) * x_m
                         w_m = softmax_m( -||z_0_hat - x_m||^2 / (2 sigma^2) )
  Re-derive corrected v: v_corr = (z_t - z_0_hat) / sigma_t

SGF mode (ICLR 2026 oral): same kernel-based repellency but applied to
*velocity directly* in flow-matching space (negative guidance on flow).
"""
import os, sys, json, argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline

# Reuse SAFREE token / latent filter helpers from existing script.
sys.path.insert(0, '/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG')
from generate_flux1_safree import (
    safree_token_projection, t5_encode_concepts, clip_encode_texts,
    encode_prompt_flux1, t5_encode_masked_prompt, clip_encode_masked_prompt,
    latent_safree_filter, compute_svf_beta, set_seed, load_prompts,
    DEFAULT_TARGET_CONCEPTS,
)


# -----------------------------------------------------------------------------
# Repellency module (RBF kernel on flat latents)
# -----------------------------------------------------------------------------
class FluxRepellency:
    """Push z_0_hat away from negative-data subspace via RBF-weighted reference subtraction."""
    def __init__(self, ref_latents: torch.Tensor, scale: float = 0.5,
                 sigma: float = 1.0, t_low: float = 0.0, t_high: float = 0.8,
                 mode: str = 'safedenoiser'):
        # ref_latents: (M, 16, H, W) on GPU, float
        self.ref = ref_latents.float()
        self.M, self.C, self.H, self.W = ref_latents.shape
        self.ref_flat = self.ref.reshape(self.M, -1)
        # cache norm-squared for numerical-stable kernel
        self._ref_norm_sq = (self.ref_flat ** 2).sum(dim=1, keepdim=True)
        self.scale = scale
        self.sigma = sigma
        self.t_low, self.t_high = t_low, t_high
        self.mode = mode

    def in_window(self, t_norm: float) -> bool:
        return self.t_low <= t_norm <= self.t_high

    @torch.no_grad()
    def negative_score(self, z_0_hat: torch.Tensor) -> torch.Tensor:
        """Compute weighted negative-data subspace projection of z_0_hat.
        Returns same shape as z_0_hat."""
        x_flat = z_0_hat.reshape(z_0_hat.shape[0], -1).float()  # (B, F)
        # Distance^2: ||x||^2 + ||r||^2 - 2 x.r
        x_norm_sq = (x_flat ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        cross = x_flat @ self.ref_flat.T                    # (B, M)
        dist2 = x_norm_sq + self._ref_norm_sq.T - 2.0 * cross  # (B, M)
        # softmax kernel weights
        log_w = -dist2 / (2.0 * self.sigma ** 2)
        w = torch.softmax(log_w, dim=-1)        # (B, M)
        neg = w @ self.ref_flat                  # (B, F)
        return neg.reshape_as(z_0_hat).to(z_0_hat.dtype)

    @torch.no_grad()
    def apply_to_z0(self, z_0_hat: torch.Tensor) -> torch.Tensor:
        """SafeDenoiser-style: subtract weighted ref from z_0_hat."""
        neg = self.negative_score(z_0_hat)
        return z_0_hat - self.scale * neg

    @torch.no_grad()
    def apply_to_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """SGF-style: directly perturb velocity in flow direction away from neg subspace."""
        neg_v = self.negative_score(v)
        return v - self.scale * neg_v


# -----------------------------------------------------------------------------
# Main generation loop with repellency hook
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_with_repellency(pipe, prompt: str, args, repellency: FluxRepellency, gen):
    """One-prompt FLUX generation with repellency intercept on each scheduler step."""
    device = pipe._execution_device
    height, width = args.height, args.width
    num_steps = args.steps

    # Encode prompts (CLIP-L pooled + T5 sequence)
    pooled, t5 = encode_prompt_flux1(pipe, prompt, device, args.max_sequence_length)

    # SAFREE token filter (optional)
    if args.safree_token_filter:
        target_t5 = t5_encode_concepts(pipe, args.target_concepts, device, args.max_sequence_length)
        target_clip = clip_encode_texts(pipe, args.target_concepts, device)
        prompt_clip = clip_encode_masked_prompt(pipe, prompt, device)
        # token-level projection (keep API of safree)
        t5 = safree_token_projection(t5, target_t5, alpha=args.safree_alpha)

    # Set up scheduler
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    # Latent
    latent = torch.randn((1, 16, height // 8, width // 8), device=device,
                         dtype=torch.bfloat16, generator=gen)
    latent = pipe.scheduler.scale_noise(latent, torch.tensor([sigmas[0]], device=device), latent)

    for i, t in enumerate(timesteps):
        # FLUX expects packed sequence [B, seq_len, 64]
        packed = pipe._pack_latents(latent, 1, 16, height // 8, width // 8)

        # Velocity prediction
        v = pipe.transformer(
            hidden_states=packed,
            encoder_hidden_states=t5,
            pooled_projections=pooled,
            timestep=t.expand(1) / 1000.0,
            return_dict=False,
        )[0]
        v = pipe._unpack_latents(v, height, width, 8)
        # Apply CFG-equivalent guidance scale (FLUX-dev uses guidance embedding, no real CFG)

        # ----- repellency intercept -----
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1] if (i + 1) < len(sigmas) else torch.tensor(0.0, device=device)
        t_norm = float(sigma_t.item())  # in [0, 1] for flow matching
        if repellency is not None and repellency.in_window(t_norm):
            if repellency.mode == 'sgf':
                v = repellency.apply_to_velocity(v)
            else:
                # SafeDenoiser: derive z_0 estimate, apply, then re-derive v
                z_0_hat = latent - sigma_t * v
                z_0_corr = repellency.apply_to_z0(z_0_hat)
                v = (latent - z_0_corr) / max(sigma_t.item(), 1e-6)
        # --------------------------------

        # Step
        latent = latent + (sigma_next - sigma_t) * v

    # Decode
    latent = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(latent.to(pipe.vae.dtype), return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type='pil')[0]
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='black-forest-labs/FLUX.1-dev')
    p.add_argument('--prompts', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--ref_latents', required=True,
                   help='Path to .pt with (M, 16, H, W) negative reference latents')
    p.add_argument('--nsamples', type=int, default=1)
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--guidance_scale', type=float, default=3.5)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--max_sequence_length', type=int, default=512)
    p.add_argument('--start_idx', type=int, default=0)
    p.add_argument('--end_idx', type=int, default=-1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16'])

    # Repellency
    p.add_argument('--mode', default='safedenoiser', choices=['safedenoiser', 'sgf'],
                   help='safedenoiser=push z_0_hat off neg subspace; sgf=push velocity off')
    p.add_argument('--repellency_scale', type=float, default=0.5)
    p.add_argument('--repellency_sigma', type=float, default=1.0,
                   help='RBF kernel bandwidth (in latent flat-norm units)')
    p.add_argument('--repellency_t_low', type=float, default=0.0)
    p.add_argument('--repellency_t_high', type=float, default=0.8)

    # SAFREE knobs (passed through)
    p.add_argument('--target_concepts', nargs='+', default=DEFAULT_TARGET_CONCEPTS)
    p.add_argument('--safree_alpha', type=float, default=0.01)
    p.add_argument('--safree_token_filter', action='store_true')
    p.add_argument('--safree_re_attention', action='store_true')
    p.add_argument('--safree_latent_filter', action='store_true')
    p.add_argument('--safree_latent_strength', type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    print(f'[FLUX-Repellency] mode={args.mode} scale={args.repellency_scale} '
          f'sigma={args.repellency_sigma} window=[{args.repellency_t_low},{args.repellency_t_high}]')
    print(f'  ref_latents: {args.ref_latents}')

    # Load pipeline
    pipe = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load repellency
    ref_latents = torch.load(args.ref_latents, map_location=device).float()
    print(f'  ref_latents shape: {tuple(ref_latents.shape)}')
    repellency = FluxRepellency(ref_latents,
                                scale=args.repellency_scale,
                                sigma=args.repellency_sigma,
                                t_low=args.repellency_t_low,
                                t_high=args.repellency_t_high,
                                mode=args.mode)

    # Prompts
    prompts = load_prompts(args.prompts)
    s, e = args.start_idx, (len(prompts) if args.end_idx < 0 else args.end_idx)
    prompts = prompts[s:e]
    os.makedirs(args.outdir, exist_ok=True)

    # Save args
    json.dump(vars(args), open(f'{args.outdir}/args.json', 'w'), indent=2, default=str)

    stats = []
    for pi, prompt in enumerate(prompts):
        seed = args.seed + (s + pi) * args.nsamples
        gen = torch.Generator(device=device).manual_seed(seed)
        set_seed(seed)
        for si in range(args.nsamples):
            img = generate_with_repellency(pipe, prompt, args, repellency, gen)
            fname = f'{(s+pi):04d}_{si:02d}.png'
            img.save(f'{args.outdir}/{fname}')
            stats.append({'pi': s + pi, 'si': si, 'seed': seed, 'prompt': prompt[:120]})
            print(f'  [{s+pi+1}/{e}] saved {fname}')

    json.dump(stats, open(f'{args.outdir}/generation_stats.json', 'w'), indent=2)
    print('Done!')


if __name__ == '__main__':
    main()
